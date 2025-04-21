import os
import torch
import torchaudio
import re
import faster_whisper

from settings.config import logger
from models.helpers import (find_numeral_symbol_tokens,
                     create_config,
                     get_words_speaker_mapping,
                     get_sentences_speaker_mapping,
                     get_realigned_ws_mapping_with_punctuation,
                     get_speaker_aware_transcript,
                     write_srt,
                     cleanup,
                     punct_model_langs,
                     langs_to_iso)

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,)


def transcribe_audio(audio_path:str,
                     enable_stemming:bool=True,
                     whisper_model_name:str="large-v3",
                     suppress_numerals:bool=True,
                     batch_size:int=8,
                     language:str="en",
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):

    '''выделению вокала из остального аудио'''
    if enable_stemming:
        return_code = os.system(f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs" --device "{device}"')

        if return_code != 0:
            logger.warning("Source splitting failed, using original audio file.")
            vocal_target = audio_path
        else:
            vocal_target = os.path.join("temp_outputs",
                                        "htdemucs",
                                        os.path.splitext(os.path.basename(audio_path))[0],
                                        "vocals.wav",)
    else:
        vocal_target = audio_path

    '''Расшифровка звука с использованием шепота и изменение временных меток с помощью принудительного выравнивания'''
    whisper_model = faster_whisper.WhisperModel(
        whisper_model_name,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(audio_waveform,
                                                                language,
                                                                suppress_tokens=suppress_tokens,
                                                                batch_size=batch_size,
                                                                without_timestamps=True,)

    else:
        transcript_segments, info = whisper_model.transcribe(audio_waveform,
                                                             language,
                                                             suppress_tokens=suppress_tokens,
                                                             without_timestamps=True,
                                                             vad_filter=True,)

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # Очистка оперативы GPU
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    '''Приведение транскрипции в соответствие с исходным звуком с помощью принудительного выравнивания'''
    alignment_model, alignment_tokenizer = load_alignment_model(device,
                                                                dtype=torch.float16 if device == "cuda" else torch.float32)

    audio_waveform = (
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device))

    emissions, stride = generate_emissions(alignment_model,
                                           audio_waveform,
                                           batch_size=batch_size)

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(full_transcript,
                                                   romanize=True,
                                                   language=langs_to_iso[info.language],)

    segments, scores, blank_token = get_alignments(emissions,
                                                   tokens_starred,
                                                   alignment_tokenizer,)

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    '''Преобразование аудио в моно для обеспечения совместимости с NeMo'''
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(os.path.join(temp_path, "mono_file.wav"),
                    audio_waveform.cpu().unsqueeze(0).float(),
                    16000,
                    channels_first=True,)

    '''Запись дневника диктора с использованием модели NeMo MSDD'''
    # msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cpu")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
    msdd_model.diarize()
    print(13)
    del msdd_model
    torch.cuda.empty_cache()

    '''Сопоставление говорящих с предложениями в соответствии с временными метками'''
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    '''Корректировка речевых фрагментов с использованием знаков препинания'''
    if info.language in punct_model_langs:
        # восстановление пунктуации в стенограмме, чтобы помочь перестроить предложения
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logger.warning(f"Punctuation restoration is not available for {info.language} language. Using the original punctuation.")

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    '''Сохранение результата модели'''
    transcript = get_speaker_aware_transcript(ssm)
    cleanup(temp_path)
    logger.info('Готово! Полученный результат находится в "results"')
    return transcript


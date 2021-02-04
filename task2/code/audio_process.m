function audio_processed = audio_process(audio_raw)
    fs = 20000; % 采样率
    Fpass = 1700; Fstop = 2200;
    ft = voice_filter(Fpass, Fstop);
    audio_raw = filter(ft, audio_raw);
    audio_processed = spec_sub(audio_raw, fs);
end
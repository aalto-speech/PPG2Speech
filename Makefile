all: data_prep speaker_emb ppg log_f0 stats tts_train tts_inference

features: speaker_emb ppg log_f0 stats

data_prep:
	./scripts/perso_data.sh $(perso_dir) $(data_dir)

speaker_emb:
	python -m ppg_tts.feature_extract.spk_emb_extract --data_dir $(data_dir) --device $(device) \
	--auth_token $(auth_token)

ppg:
	python -m ppg_tts.feature_extract.ppg_extract --data_dir $(data_dir) --device $(device) --no_ctc

log_f0:
	python -m ppg_tts.feature_extract.log_f0_extract --data_dir $(data_dir)

stats:
	python -m ppg_tts.feature_extract.make_stats --data_dir $(data_dir)

tts_train:
	python -m ppg_tts.main fit -c $(config)

tts_inference:
	python -m ppg_tts.main predict -c $(config)

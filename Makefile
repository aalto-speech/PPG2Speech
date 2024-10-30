all: data_prep speaker_emb ppg tts_train tts_inference

data_prep:
	./scripts/perso_data.sh $(perso_dir) $(data_dir)

speaker_emb:
	python -m ppg_tts.feature_extract.spk_emb_extract --data_dir $(data_dir) --device $(device) \
	--auth_token $(auth_token)

ppg:
	python -m ppg_tts.feature_extract.ppg_extract --data_dir $(data_dir) --device $(device)

tts_train:
	python -m ppg_tts.train

tts_inference:
	python -m ppg_tts.inference

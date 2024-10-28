all: data_prep speaker_emb ppg tts_train tts_inference

data_prep:
	./scripts/perso_data.sh $(perso_dir) $(data_dir)

speaker_emb:
	python -m ppg_tts.speaker_emb.extract --data_dir $(data_dir)

ppg:
	python -m ppg_tts.ppg.extract --data_dir $(data_dir)

tts_train:
	python -m ppg_tts.train

tts_inference:
	python -m ppg_tts.inference

import random
import sys
from collections import defaultdict
import wespeaker
from loguru import logger


def load_wav_scp(wav_scp_path):
    """Load wav.scp file and group utterances by speaker."""
    speaker_utterances = defaultdict(list)
    with open(wav_scp_path, 'r') as f:
        for line in f:
            key, path = line.strip().split()
            speaker_id = key.split('_')[0]  # Extract speaker ID
            speaker_utterances[speaker_id].append(path)
    return speaker_utterances


if __name__ == "__main__":
    scp = sys.argv[1]

    # Load wav.scp file
    speaker_utterances = load_wav_scp(scp)

    # Load the speaker embedding model
    SpEmModel = wespeaker.load_model('vblinkf')
    avg_simi = 0.0
    num_comparisons = 0

    for speaker_id, utterances in speaker_utterances.items():
        # Skip speakers with fewer than 2 utterances
        if len(utterances) < 2:
            logger.warning(f"Speaker {speaker_id} has less than 2 utterances. Skipping.")
            continue

        # Randomly select two utterances for comparison
        for _ in range(20):
            utterance1, utterance2 = random.sample(utterances, 2)

            # Compute similarity
            similarity = SpEmModel.compute_similarity(utterance1, utterance2)

            num_comparisons += 1
            avg_simi += (similarity - avg_simi) / num_comparisons

    logger.info(f"The average cosine similarity across all speakers is {avg_simi:.4f}")
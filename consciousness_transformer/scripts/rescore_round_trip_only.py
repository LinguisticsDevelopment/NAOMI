"""Standalone round-trip eval: load the already-trained encoder_colab.pt and
decoder_colab.pt checkpoints (NO training here) and run the autoencoder
round-trip (text -> encoder.beam_decode -> top tree -> decoder.realize ->
text) on N held-out English test records, reusing
colab_train_all.evaluate_autoencoder_round_trip / predicted_tree_to_structure
verbatim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from nsm_ct import decoder_trained as dt
from train_encoder import load_gold, stratified_split
import colab_train_all as cta

ROOT = Path(__file__).resolve().parent.parent

def main():
    enc_ckpt = torch.load(ROOT / "runs" / "encoder_colab.pt", map_location="cpu", weights_only=False)
    dec_ckpt = torch.load(ROOT / "runs" / "decoder_colab.pt", map_location="cpu", weights_only=False)

    pos_vocab = enc_ckpt["pos_vocab"]
    role_vocab = enc_ckpt["role_vocab"]
    encoder = em.EncoderModel(pos_vocab, role_vocab, d_axes=enc_ckpt["d_axes"],
                               hash_buckets=enc_ckpt["hash_buckets"], d_model=enc_ckpt["d_model"],
                               controller_hidden=enc_ckpt["d_model"])
    encoder.load_state_dict(enc_ckpt["model_state"])
    encoder.eval()

    decoder = dt.DecoderTrainedModel(dec_ckpt["relation_vocab"], dec_ckpt["function_vocab"],
                                      hash_buckets=dec_ckpt["hash_buckets"], d_model=dec_ckpt["d_model"])
    decoder.load_state_dict(dec_ckpt["model_state"])
    decoder.eval()

    usvs = load_usvs(str(ROOT / "data" / "usvs"))

    cfg = enc_ckpt["config"]
    records = load_gold(str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    train_recs, dev_recs, test_recs = stratified_split(records, cfg["seed"], cfg["n_train"],
                                                         cfg["n_dev"], cfg["n_test"])
    round_trip_records = test_recs[:15]
    print(f"round-trip test on {len(round_trip_records)} held-out English test-split sentences "
          f"(encoder test split, n={len(test_recs)} total)")

    metrics = cta.evaluate_autoencoder_round_trip(
        encoder, decoder, round_trip_records, usvs, pos_vocab, enc_ckpt["hash_buckets"],
        beam_width=8, k=8, log=print)

    print("\n" + "=" * 78)
    print("ROUND-TRIP RESULT (encoder.beam_decode -> decoder.realize, fixed beam_decode)")
    print("=" * 78)
    print(f"n={metrics['n']}  empty_forest={metrics['empty_forest']}")
    print(f"exact_match={metrics['exact_match']:.3f}  token_f1={metrics['token_f1']:.3f}")
    print("\n5 gold-vs-realized example pairs:")
    for ex in metrics["examples"]:
        print(f"  GOLD: {ex['gold']!r}")
        print(f"  PRED: {ex['pred']!r}")
        print()


if __name__ == "__main__":
    main()

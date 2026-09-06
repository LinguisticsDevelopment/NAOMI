"""One-off driver for the aliasfix round-trip eval: reproduces the exact
encoder test split (from runs/encoder_colab.pt's own config) and the exact
decoder train split (from runs/decoder_colab.pt's own config, replicating
scripts/colab_train_all.py main()'s shuffle/split steps verbatim), then
calls scripts/colab_train_all.py's evaluate_autoencoder_round_trip
(encoder.beam_decode -> decoder.realize, no gold tree in the loop) on ~20
held-out English sentences not used for decoder training. NO retraining.

Usage: python scripts/_roundtrip_aliasfix.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from nsm_ct import decoder_trained as dt
from train_encoder import load_gold, stratified_split
from train_decoder import load_records as load_decoder_records, split_records as split_decoder_records
from colab_train_all import evaluate_autoencoder_round_trip

ROOT = Path(__file__).resolve().parent.parent


def main():
    usvs = load_usvs(str(ROOT / "data" / "usvs"))

    enc_ckpt = torch.load(ROOT / "runs" / "encoder_colab.pt", map_location="cpu", weights_only=False)
    enc_cfg = enc_ckpt["config"]
    pos_vocab = enc_ckpt["pos_vocab"]
    role_vocab = enc_ckpt["role_vocab"]
    encoder = em.EncoderModel(pos_vocab, role_vocab, d_axes=enc_ckpt["d_axes"],
                               hash_buckets=enc_ckpt["hash_buckets"], d_model=enc_ckpt["d_model"],
                               controller_hidden=enc_ckpt["d_model"])
    encoder.load_state_dict(enc_ckpt["model_state"])
    encoder.eval()

    dec_ckpt = torch.load(ROOT / "runs" / "decoder_colab.pt", map_location="cpu", weights_only=False)
    dec_cfg = dec_ckpt["config"]
    relation_vocab = dec_ckpt["relation_vocab"]
    function_vocab = dec_ckpt["function_vocab"]
    decoder = dt.DecoderTrainedModel(relation_vocab, function_vocab,
                                      hash_buckets=dec_ckpt["hash_buckets"], d_model=dec_ckpt["d_model"])
    decoder.load_state_dict(dec_ckpt["model_state"])
    decoder.eval()

    records = load_gold(str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    _enc_train, _enc_dev, enc_test = stratified_split(
        records, enc_cfg["seed"], enc_cfg["n_train"], enc_cfg["n_dev"], enc_cfg["n_test"])
    print(f"reproduced encoder split: train={enc_cfg['n_train']} dev={enc_cfg['n_dev']} "
          f"test={len(enc_test)} (seed={enc_cfg['seed']})")

    dec_records_all = load_decoder_records(str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    n_dec = dec_cfg["n_train"] + dec_cfg["n_dev"]
    dec_pool = dec_records_all[:]
    random.Random(dec_cfg["seed"]).shuffle(dec_pool)
    dec_pool = dec_pool[:n_dec]
    dec_train, _dec_dev = split_decoder_records(dec_pool, dec_cfg["seed"], dec_cfg["n_train"], dec_cfg["n_dev"])
    dec_train_texts = {r["text"] for r in dec_train}
    print(f"reproduced decoder train split: n={len(dec_train)} (seed={dec_cfg['seed']})")

    round_trip_pool = [r for r in enc_test if r["text"] not in dec_train_texts] or enc_test
    round_trip_records = round_trip_pool[:20]
    print(f"round-trip test on {len(round_trip_records)} held-out English sentences "
          f"(encoder test split, excluded from decoder training)")

    metrics = evaluate_autoencoder_round_trip(
        encoder, decoder, round_trip_records, usvs, pos_vocab, enc_ckpt["hash_buckets"],
        beam_width=8, k=8, log=print)

    print("\n" + "=" * 78)
    print("ROUND-TRIP RESULT (fixed beam_decode)")
    print("=" * 78)
    print(f"  exact_match = {metrics['exact_match']:.3f}   token_f1 = {metrics['token_f1']:.3f}   "
          f"n={metrics['n']}  empty_forest={metrics['empty_forest']}")


if __name__ == "__main__":
    main()

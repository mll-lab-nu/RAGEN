#!/usr/bin/env python3
"""
Dense-only retrieval server for Search training.
Provides E5 embeddings + FAISS dense indexing.

Usage:
    python server.py --data_dir ./search_data/prebuilt_indices --port 8000
"""

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import torch
import faiss
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer


class LocalRetriever:
    """Dense-only retrieval system using FAISS."""

    def __init__(self, data_dir: str, device: str = "cpu", gpu_memory_limit_mb: int = 6144):
        self.data_dir = Path(data_dir)
        self.corpus = []
        self.dense_index = None
        self._lock = threading.Lock()

        if device.startswith("cuda"):
            dev_idx = int(device.split(":")[-1]) if ":" in device else 0
            fraction = gpu_memory_limit_mb / (torch.cuda.get_device_properties(dev_idx).total_memory / 1024**2)
            fraction = min(fraction, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)
            print(f"GPU memory limit: {gpu_memory_limit_mb}MB (fraction={fraction:.4f}) on cuda:{dev_idx}")

        self.encoder = SentenceTransformer("intfloat/e5-base-v2", device=device)
        print(f"E5 encoder loaded on device: {device}")

        self._load_data()

    def _load_data(self):
        """Load corpus and dense index from data directory."""
        print(f"Loading data from {self.data_dir}")

        # Load corpus
        corpus_file = self.data_dir / "corpus.json"
        with open(corpus_file) as f:
            self.corpus = json.load(f)
        print(f"Loaded corpus with {len(self.corpus)} documents")

        # Load dense index
        dense_index_file = self.data_dir / "e5_Flat.index"
        self.dense_index = faiss.read_index(str(dense_index_file))
        print(f"Loaded dense index with {self.dense_index.ntotal} vectors")

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Dense retrieval using FAISS. Lock serializes GPU encoding to prevent memory bloat."""
        with self._lock:
            query_vector = self.encoder.encode([f"query: {query}"]).astype("float32")
        scores, indices = self.dense_index.search(query_vector, k)

        return [{"content": self.corpus[idx], "score": float(score)} for score, idx in zip(scores[0], indices[0], strict=False) if idx < len(self.corpus)]


# Flask app
app = Flask(__name__)
retriever = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "corpus_size": len(retriever.corpus), "index_type": "dense_only", "index_loaded": retriever.dense_index is not None})


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Main retrieval endpoint."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request"}), 400

        query = data["query"]
        k = data.get("top_k", data.get("k", 10))

        results = retriever.search(query=query, k=k)

        formatted_results = [{"id": f"doc_{i}", "content": result["content"], "score": result["score"]} for i, result in enumerate(results, 1)]

        return jsonify({"query": query, "method": "dense", "results": formatted_results, "num_results": len(formatted_results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Dense-only retrieval server")
    parser.add_argument("--data_dir", default="./search_data/prebuilt_indices", help="Directory containing corpus and dense index")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--device", default="cpu", help="Device for E5 encoder (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--gpu_memory_limit_mb", type=int, default=1024, help="Max GPU memory in MB for E5 encoder (default: 1024)")

    args = parser.parse_args()

    # Initialize retriever
    global retriever
    try:
        retriever = LocalRetriever(args.data_dir, device=args.device, gpu_memory_limit_mb=args.gpu_memory_limit_mb)
        print(f"Dense retrieval server initialized with {len(retriever.corpus)} documents")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    # Start server
    print(f"Starting dense retrieval server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

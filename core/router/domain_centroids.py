
"""
Tier-2 Semantic Centroid Vector Router
Latency Budget: < 10.0 ms
Om's Research Pillar: Semantic Embedding Similarity against Domain Centroids
"""

import hashlib
import math
import re
from typing import Dict, List, Tuple
import numpy as np

# Curated High-Dimensional Semantic Anchors for Each Domain
DOMAIN_VOCABULARIES = {
    "mathematics": [
        "calculate", "compute", "solve", "equation", "formula", "integral", "derivative", "differential",
        "matrix", "vector", "eigenvalue", "determinant", "probability", "statistics", "algebra",
        "geometry", "trigonometry", "calculus", "polynomial", "theorem", "proof", "arithmetic",
        "prime", "logarithm", "exponential", "variance", "standard deviation", "permutation",
        "combination", "modulo", "fibonacci", "graph theory", "topology", "simplex", "limit",
        "infinity", "discrete", "recurrence", "linear algebra", "tensor", "complex number"
    ],
    "medical": [
        "symptom", "symptoms", "diagnosis", "diagnoses", "disease", "patient", "clinical", "treatment", "therapy", "syndrome",
        "pathology", "etiology", "pathophysiology", "pharmacology", "dosage", "medication", "drug", "prescription",
        "infection", "bacterial", "viral", "cardiovascular", "hypertension", "diabetes", "insulin", "pain", "acute",
        "glucose", "blood pressure", "oncology", "cancer", "tumor", "biopsy", "prognosis", "epigastric", "lipase",
        "adverse effect", "contraindication", "contraindications", "surgery", "anatomy", "physiology", "hematology",
        "neurology", "pediatrics", "immunology", "vaccine", "antibody", "respiratory", "renal", "pancreatitis"
    ],
    "code": [
        "code", "programming", "software", "function", "class", "method", "variable", "algorithm",
        "bug", "debug", "error", "exception", "compiler", "runtime", "python", "javascript", "typescript",
        "c++", "rust", "database", "sql", "api", "rest", "backend", "frontend", "git", "repository",
        "refactor", "optimization", "data structure", "array", "binary tree", "hash map", "sorting",
        "recursion", "async", "await", "multithreading", "docker", "unit test", "framework", "library"
    ],
    "science": [
        "physics", "chemistry", "biology", "science", "experiment", "hypothesis", "theory", "law",
        "thermodynamics", "entropy", "enthalpy", "kinetics", "atom", "molecule", "electron", "proton",
        "neutron", "quantum", "relativity", "gravity", "electromagnetism", "optics", "astronomy",
        "planet", "galaxy", "cosmology", "photosynthesis", "cell", "dna", "rna", "evolution",
        "organism", "ecosystem", "periodic table", "element", "covalent", "ionic", "speed of light"
    ],
    "writing": [
        "write", "essay", "poem", "poetry", "story", "novel", "creative", "narrative", "character",
        "plot", "theme", "metaphor", "simile", "alliteration", "tone", "style", "paraphrase",
        "rewrite", "proofread", "edit", "critique", "dialogue", "speech", "article", "blog",
        "copywriting", "headline", "persuasive", "rhetoric", "introduction", "conclusion", "stanza"
    ],
    "general": [
        "what", "why", "how", "who", "when", "where", "explain", "tell", "opinion", "think",
        "recommend", "compare", "difference", "advice", "overview", "summary", "general", "history",
        "culture", "weather", "travel", "philosophy", "lifestyle", "hobby", "question", "help"
    ]
}

# Build a compact unified vocabulary index for ultra-fast centroid mapping
ALL_TERMS = []
TERM_TO_IDX = {}
for domain, terms in DOMAIN_VOCABULARIES.items():
    for term in terms:
        t = term.lower()
        if t not in TERM_TO_IDX:
            TERM_TO_IDX[t] = len(ALL_TERMS)
            ALL_TERMS.append(t)

VOCAB_DIM = len(ALL_TERMS)
HASH_BUCKETS = 64
TOTAL_DIM = VOCAB_DIM + HASH_BUCKETS

# Construct normalized centroid vectors for each domain
DOMAIN_CENTROIDS: Dict[str, np.ndarray] = {}
for domain, terms in DOMAIN_VOCABULARIES.items():
    vec = np.zeros(TOTAL_DIM, dtype=np.float32)
    for term in terms:
        idx = TERM_TO_IDX[term.lower()]
        vec[idx] = 2.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    DOMAIN_CENTROIDS[domain] = vec


def tokenize_query(query: str) -> List[str]:
    """Extract clean alphanumeric tokens, bigrams, and trigrams."""
    cleaned = re.sub(r"[^\w\s\+\-\*\/]", " ", query.lower())
    tokens = cleaned.split()
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]
    return tokens + bigrams + trigrams



def compute_query_vector(query: str) -> np.ndarray:
    """
    Projects text into normalized hybrid semantic-lexical space.
    Combines domain anchor centroids with deterministic open-vocabulary hashing.
    """
    tokens = tokenize_query(query)
    vec = np.zeros(TOTAL_DIM, dtype=np.float32)

    for t in tokens:
        if t in TERM_TO_IDX:
            vec[TERM_TO_IDX[t]] += 2.0

        # Deterministic open-vocabulary hash feature
        hash_bytes = hashlib.sha256(t.encode("utf-8")).digest()[:8]
        hash_value = int.from_bytes(hash_bytes, byteorder="big")
        h_idx = VOCAB_DIM + (hash_value % HASH_BUCKETS)

        vec[h_idx] += 1.0

    # Normalize the vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec



def evaluate_centroids(query: str) -> Dict[str, float]:
    """
    Computes cosine similarity between the query vector and all domain centroids.
    Returns: Dict of domain -> probability score (softmax-normalized).
    Execution time: ~0.5 - 2 ms.
    """
    q_vec = compute_query_vector(query)
    
    # If the query contains no vocabulary matches, return uniform with slight general bias
    if np.linalg.norm(q_vec) == 0:
        return {
            "general": 0.35,
            "writing": 0.15,
            "science": 0.15,
            "code": 0.12,
            "mathematics": 0.12,
            "medical": 0.11
        }
    
    # Dot product since vectors are unit normalized
    raw_scores = {}
    for domain, c_vec in DOMAIN_CENTROIDS.items():
        similarity = float(np.dot(q_vec, c_vec))
        raw_scores[domain] = max(0.0, similarity)
    
    # Softmax scaling with temperature
    temperature = 0.25
    exp_scores = {d: math.exp(s / temperature) for d, s in raw_scores.items()}
    total = sum(exp_scores.values())
    
    probabilities = {d: exp_scores[d] / total for d in exp_scores}
    return probabilities

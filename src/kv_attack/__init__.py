# ── Server ───────────────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_HOST     = "localhost"
VLLM_PORT     = 8001

# Primary model — DeepSeek-R1-Distill-Llama-8B is built on the Llama-3.1-8B
# architecture, so it uses the same tokenizer and BOS behaviour as Llama-3.1.
# Override at runtime with --model-id for any other cached model.
MODEL_ID      = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

BLOCK_SIZE    = 16


def detect_has_bos(model_id: str) -> bool:
    """
    Return True if vLLM prepends a BOS token for this model family.

    This affects block alignment: build_aligned_system_prompt() adds a 1-token
    BOS offset when computing how many tokens fit in prefix blocks.

    Llama-family models (including DeepSeek-R1-Distill-Llama-*):
        vLLM prepends <|begin_of_text|> (token id 128000) → has_bos = True

    Qwen2-family models (Qwen2.5-*):
        vLLM does NOT prepend a BOS token for Qwen2 → has_bos = False
        Using has_bos=True for Qwen shifts all private blocks by 1 token,
        breaks alignment, and degrades the timing oracle.

    Add new families here as they are tested.
    """
    mid = model_id.lower()
    if any(k in mid for k in ("qwen2", "qwen2.5", "qwen-2")):
        return False
    # Llama-3, DeepSeek-R1-Distill-Llama, Mistral, etc. all use BOS
    return True

# ── Timing parameters ─────────────────────────────────────────────────────────
N_REPEATS_FAST    = 1   # Stage 2 probes: full-hit vs S1-HIT gap is ~175 ms >> noise
N_REPEATS_STAGE1  = 3   # Stage 1 probes: S1-HIT vs MISS gap is smaller; average 3 samples
                        # to reduce false-positive rate. Cost: 3× Stage 1 API calls.
N_REPEATS_CONFIRM = 3
N_TOP_CANDIDATES  = 3
N_CALIBRATION     = 200
KS_ALPHA          = 1e-8

# ── KV-cache eviction parameters ─────────────────────────────────────────────
KV_CACHE_BLOCKS = 1024
EVICT_REQUESTS  = 100
EVICT_TOKENS    = 220

# ── Self-eviction prevention ──────────────────────────────────────────────────
RESEED_EVERY = 4

# ── Medical domain vocabulary ─────────────────────────────────────────────────
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert",
    "Jennifer", "Michael", "Linda", "William", "Barbara",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Wilson", "Martinez",
]
MEDICAL_CONDITIONS = [
    "diabetes", "hypertension", "asthma", "arthritis", "depression",
    "anxiety", "COPD", "obesity", "hypothyroidism", "hyperlipidemia",
    "coronary artery disease", "chronic kidney disease", "heart failure",
    "atrial fibrillation", "osteoporosis", "Parkinson's disease",
    "multiple sclerosis", "epilepsy", "migraine", "sleep apnea",
]
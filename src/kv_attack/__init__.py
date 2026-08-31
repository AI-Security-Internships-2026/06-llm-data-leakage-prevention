
# ── Server ───────────────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_HOST     = "localhost"
VLLM_PORT     = 8001
MODEL_ID      = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BLOCK_SIZE    = 16

# ── Timing parameters ─────────────────────────────────────────────────────────
N_REPEATS_FAST    = 1
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

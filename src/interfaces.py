"""
Shared interface contract between model (Person A) and data/training (Person B).

DO NOT change these shapes or conventions without both parties agreeing.
"""

# Variable order — fixed, used by both model and dataset
VARIABLES = ["SST", "HC", "taux", "tauy"]
NUM_VARIABLES = len(VARIABLES)  # 4

# === Model Interface ===
#
# Forward signature:
#   model(x, target_month, rollout_step=None) -> y
#
# Args:
#   x: (B, L, 4, H, W)  float32, normalized input fields
#   target_month: (B,)   int64, values in [1, 12]
#   rollout_step: (B,)   int64, 0=single-step, 1/2/3...=rollout step
#
# Returns:
#   y: (B, 1, 4, H, W)  float32, predicted next-month fields (same normalization as input)
#
# === Dataset Interface ===
#
# Each sample returns:
#   x: (L, 4, H, W)           float32, input window
#   y: (1, 4, H, W)           float32, target month
#   target_month: int          1-12
#   (rollout_step is handled at inference time, not in training)

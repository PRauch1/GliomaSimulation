KEY_PATIENT_ID = "patient_id"
KEY_OP_ID = "op_id"
KEY_BASELINE = "baseline"
KEY_IMAGE = "image"
KEY_BASELINE_MASK = "residual_mask"
KEY_FOLLOWUPS = "followups"
KEY_MASK = "mask"
KEY_ROOT = "meta_root"
KEY_BASELINE_IMG = "baseline_img"
KEY_FOLLOWUPS_IMG = "followup_img"
KEY_FOLLOWUPS_MASK = "followup_mask"
KEY_DAY = "day"
KEY_TS = "timestamp"
KEY_MODALITY = "modality"
KEY_GEOMETRY = "geometry"
KEY_CONVENTION = "convention"
KEY_SPACING = "spacing_mm"
KEY_SIZE = "size"
KEY_ORIGIN = "origin"
KEY_DIRECTION = "direction"
KEY_CROP = "crop"
KEY_IDX = "index"

PAD_MULTIPLE = 8
CHANNELS = (16, 32, 64, 128)

FILM_CORTEX = True
FILM_IDH = True
FILM_HISTO = True

FILE_FILM_CSV = "film.csv"
FILE_PRED_PROB_RAW = "pred_prob_raw"
FILE_PRED_PROB_ENF = "pred_prob_enf"
FILE_PRED_MASK_RAW = "pred_mask_raw"
FILE_PRED_MASK_ENF = "pred_mask_enf"

FILM_KEY_CORTEX = "cortex"
FILM_KEY_IDH = "idh"
FILM_KEY_HISTO = "histo"
FILM_KEY_UNK = "__UNK__"

META_FILE_NAME = "meta.json"
MODEL_FILE_NAME = "best_unet_film.pth"
DIR_MODEL = "checkpoints"
DIR_EVAL_OUTPUTS = "eval_outputs_film"
DIR_DATASET = "dataset"
DIR_GLOBAL_EVAL = "GLOBAL_EVAL"
DIR_CURVES_RAW = "curves_raw"
DIR_CURVES_ENF = "curves_enf"
DIR_VALIDATION = "validation/UCSF"

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_VOX_SAMPLES = 200_000
CALIBRATION_NBINS = 15
NSD_TOLS_MM = (1.0, 2.0)
MIN_MASK_VOXELS = 10

CKPT_STATE_DICT = "state_dict"
CKPT_MAX_DAYS = "t_max_days"
CKPT_PAD_MULTIPLE = "pad_multiple"
CKPT_CHANNELS = "channels"
CKPT_SEED = "seed"
CKPT_MODEL = "model"
CKPT_DEVICE = "device"
CKPT_ADD_LOOKUP = "add_lookup"
CKPT_VOCABS = "vocabs"
CKPT_META_BUNDLE = "meta_bundle"
CKPT_FILM = "film_enabled"
CKPT_ENABLED = "enabled"
CKPT_META_CONFIG = "meta_cfg"
CKPT_META_EMB_DIM = "meta_emb_dim"
CKPT_META_DIM = "meta_dim"
CKPT_META_DROPOUT = "meta_dropout"
CKPT_CAT_ORDER = "cat_order"
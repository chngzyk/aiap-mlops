import aiap_mlops as ops
import aiap_mlops_fastapi as ops_fapi


PRED_MODEL = ops.modeling.utils.load_model(
    ops_fapi.config.SETTINGS.PRED_MODEL_PATH)

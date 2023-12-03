from typing import Optional

from pydantic import BaseModel


class ValidationResultsModel(BaseModel):
    roc_auc: Optional[str] = None
    roc_auc_lof: Optional[str] = None
    roc_auc_vae: Optional[str] = None
    roc_auc_1nn: Optional[str] = None
    roc_auc_ano_gan: Optional[str] = None

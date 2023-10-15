from typing import Optional
from pydantic import BaseModel


class ValidationResultsModel(BaseModel):
    roc_auc_plot_one_hot: Optional[str] = None
    roc_auc_plot_ignore_labels: Optional[str] = None
    roc_auc_for_image_data: Optional[str] = None

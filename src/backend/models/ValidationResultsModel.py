from typing import Optional
from pydantic import BaseModel


class ValidationResultsModel(BaseModel):
    roc_auc_plot_one_hot: Optional[str] = None
    roc_auc_plot_one_hot_plain_mahalanobis: Optional[str] = None
    roc_auc_plot_factor_2: Optional[str] = None
    roc_auc_plot_factor_10: Optional[str] = None
    roc_auc_plot_ignore_labels: Optional[str] = None
    t_sne_plot_original_input_data: Optional[str] = None
    t_sne_plot_one_hot_weighted_data: Optional[str] = None
    t_sne_plot_one_hot_weighted_data_ignore_labels: Optional[str] = None
    t_sne_plot_weighted_data_factor_10: Optional[str] = None
    t_sne_plot_weighted_data_factor_100: Optional[str] = None

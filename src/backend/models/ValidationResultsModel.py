from pydantic import BaseModel


class ValidationResultsModel(BaseModel):
    roc_auc_plot_one_hot: str
    roc_auc_plot_factor_2: str
    roc_auc_plot_factor_10: str
    roc_auc_plot_ignore_labels: str
    t_sne_plot_original_input_data: str
    t_sne_plot_one_hot_weighted_data: str
    t_sne_plot_one_hot_weighted_data_ignore_labels: str
    t_sne_plot_weighted_data_factor_10: str
    t_sne_plot_weighted_data_factor_100: str

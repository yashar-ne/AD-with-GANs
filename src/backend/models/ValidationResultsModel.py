from pydantic import BaseModel


class ValidationResultsModel(BaseModel):
    roc_auc_plot: str
    t_sne_plot_original_input_data: str
    t_sne_plot_one_hot_weighted_data: str
    t_sne_plot_weighted_data_factor_5: str
    t_sne_plot_weighted_data_factor_10: str

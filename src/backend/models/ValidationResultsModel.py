from pydantic import BaseModel


class ValidationResultsModel(BaseModel):
    roc_auc_plot: str
    t_sne_plot_original_input_data: str

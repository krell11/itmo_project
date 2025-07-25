from typing import Dict, Optional, List, Union, Any, Literal
from pydantic import Field, BaseModel, field_validator


class ProjectConfig(BaseModel):
    project_path: str
    script_paths: Optional[List[str]] = None
    bat_path: Optional[str] = None
    results_file: Optional[str] = None
    number_of_cores: int = Field(default=1, ge=1)
    del_data: bool = False
    date: bool = True
    doe_file: Optional[str] = None
    rom_path: Optional[str] = None
    project_name: str = "project"

    class Config:
        extra = "forbid"


class ExperimentConfig(BaseModel):
    mode: str = Field(default="TRAIN", pattern="^(TRAIN|PREDICT|TP)$")
    n_snapshots: Optional[int] = None
    needed_params: Optional[List[List[float]]] = None
    doe_type: str = Field(default="LHS", pattern="^(LHS|MEPE)$")
    input_scaler: Optional[Union[str, List[str]]] = None
    output_scaler: Optional[Union[str, List[str]]] = None

    class Config:
        extra = "forbid"


class PostProcessorConfig(BaseModel):
    enabled: bool = True
    bat_path: str = Field(..., description="Абсолютный путь до исполняемого файла META")
    geometry_path: str = Field(..., description="Абсолютный путь до КЭ модели")
    output_directory: Optional[str] = None

    class Config:
        extra = "forbid"


class SVDConfig(BaseModel):
    rom_type: Literal["SVD"] = "SVD"
    rank: int = Field(default=100, ge=1)
    n_oversampling: int = Field(default=0, ge=0)
    n_power_iters: int = Field(default=3, ge=1)
    reduction: Optional[int] = None


class NNConfig(BaseModel):
    rom_type: Literal["NN"] = "NN"
    train_method: str = Field(default="lm", pattern="^(lm|sgd)$")
    mu: Optional[float] = 0.001
    learning_rate: Optional[float] = None
    n_epochs: int = Field(default=100, ge=1)
    use_auto_search: bool = False
    units: Optional[List[int]] = None
    activations: Optional[List[str]] = None
    complexity: str = Field(default="Simple", pattern="^(Simple|Medium|Complex)$")


class GPRConfig(BaseModel):
    rom_type: Literal["GPR"] = "GPR"


class VariableConfig(BaseModel):
    name: str
    bounds: Optional[List[float]] = None
    script_name: Optional[str] = None
    symbol: str = " "
    line: Optional[int] = None
    position: Optional[int] = None

    @field_validator('bounds')
    def validate_bounds(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError("Границы должны содержать ровно два значения [min, max]")
            if v[0] >= v[1]:
                raise ValueError("Минимальное значение должно быть меньше максимального")
        return v
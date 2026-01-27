from pydantic_settings import BaseSettings


class TransformationsConfig(BaseSettings):
    img_size: tuple[int, int]
    mean_normalization_values: list[float]
    std_normalization_values: list[float]

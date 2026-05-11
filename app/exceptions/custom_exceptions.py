from fastapi import HTTPException, status

class PredictionException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

class ModelNotLoadedException(HTTPException):
    def __init__(self, model_name: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Model {model_name} not loaded. Please contact administrator."
        )

class InvalidInputException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

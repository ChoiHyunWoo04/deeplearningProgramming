class EarlyStop:
    # 실패 민감도 (인내도) 설정 필요시 수정해주세요
    LIMIT_PATIENCE = 20
    
    def __init__(self) -> None:
        self.patience = 0
    
    def update_patience(self, best_loss, curr_loss) -> bool:
        if curr_loss > best_loss:
            self.__add_patience()
        else:
            self.__clear_patience()
        return self.patience < EarlyStop.LIMIT_PATIENCE
    
    def __add_patience(self) -> None:
        self.patience += 1
    
    def __clear_patience(self) -> None:
        self.patience = 0
    
    def __str__(self) -> str:
        return f"current patience : {self.patience}"
    
    def __repr__(self) -> str:
        return f"current patience : {self.patience} ( limit {EarlyStop.LIMIT_PATIENCE} )"
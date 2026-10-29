from metalmind.contracts import PromotionDecision

def decide_promotion(verification,confirmations):
    passed=verification if isinstance(verification,bool) else verification.passed
    if not passed: return PromotionDecision(promote=False,reason="correctness not established")
    if len(confirmations)!=2 or any(not session for session in confirmations):
        return PromotionDecision(promote=False,reason="two independent confirmation sessions required")
    for session in confirmations:
        if any(c.pairs<30 or c.speedup<1.05 or c.lower<=1.0 for c in session):
            return PromotionDecision(promote=False,reason="performance threshold or confidence gate not met")
    return PromotionDecision(promote=True,reason="correct and confirmed faster in both sessions")

# Deep Residual Learning for Image Recognition

## Summary
1. 네트워크는 깊을수록 성능이 좋다.
2. 깊으면 vanishing/exploding gradient가 발생하여 수렴(Convergence)하기 어렵다.
    → normalization, SGD로 해결 가능
3. 깊으면 수렴하지만 degradation이 발생하는 문제 발생
    → overfitting 문제는 아니다 (training error와 validation error가 모두 감소하기 때문)
    → 학습이 어렵기 때문이지 않을까?
4. shallow vs deep network 비교했을 때 만약 deep network가 shallow network에서 단순히 identity mapping만 추가된다면 
최소 shallow network와 성능이 같거나 높아야하는데 실험 결과 그렇지 않다.
5. 이러한 실험 결과는 identity mapping에 대해서 학습이 어렵기 때문이다. (H(x) → x) nonlinearity를 가지는 layer를 지났는데도 
입력과 같아지는 것은 힘들기 때문
6. F(x) + x → x 로 shortcut connection을 추가하면 F(x)의 weight를 0으로 만드는 방식으로 학습을 하면 identity mapping이 가능하여
쉽게 학습이 가능하다.
7. 실제로 identity mapping이 network에 필요한가에 대한 의문은 layer response 실험을 통해 F(x)가 0에 가깝게 학습이 된다는 것으로 입증

## Q&A
1. Analysis of Layer Responses 질문

   ![img.png](img.png)
   1. Analysis of Layer Responses에서 std이 작다는 것이 왜 residual function이 0에 가까워지는 것과 연관이 되는지?
   2. std가 마지막 index layer에서 커지는 이유가 무엇인가요?

2. x → H(x) 라는 identity function으로 입력과 출력이 같아야하는 경우가 있는지? (왜 굳이 identity mapping으로 이를 보여주는지?)

   * *"There exists a solution by construction to the deeper model: the added layers are identity mapping,
        and the other layers are copied from the learned shallower model. The existence of this constructed solution 
        indicates that a deeper model should produce no higher training error than its shallower counterpart."*
     * deep network는 shallow network보다 error가 높으면 안된다. 왜나햐면 deep network가 shallow network에서 layer를 몇 개 
        더 쌓는다고 했을 때, 이미 shallow network에서 가진 출력이 optimal 하다면 그 뒤 layer는 단순히 identity하게 입력과 출력이
        같도록 하는 layer로 기능을하여 가장 마지막 출력으로 넘겨주기만해도 shallow network보다 error가 떨어지지는 않을 것이기 때문이다.
        ⇒ identity mapping을 예시로 설명한 이유
   * *In real cases, it is unlikely that identity mappings are optimal, but our reformulation may help to precondition 
     the problem. If the optimal function is closer to an identity mapping than to a zero mapping, 
     it should be easier for the solver to ﬁnd the perturbations with reference to an identity mapping, 
     than to learn the function as a new one. We show by experiments (Fig. 7) that the learned residual functions 
     in general have small responses, suggesting that identity mappings provide reasonable preconditioning.*
     * 실제에서는 identity mapping을 하는 layer로 동작하는 것이 최적인 경우가 드물겠지만, 문제를 해결하는 전제 조건을 도와준다. 
     → 만약에 identity mapping이 최적일 때 도움을 준다는 의미 → 실제로 Fig.7 에서는 identity mapping이 되도록 학습 되는 것으로 보아 
     identity mapping이 최적인 경우가 있다는 것도 입증
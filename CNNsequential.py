import torch                 # PyTorch 기본 라이브러리이다.
import torch.nn as nn        # 신경망 layer들을 사용하기 위해 torch.nn을 nn이라는 이름으로 가져온다.


# LeNet-5 형태의 CNN 모델을 정의한다.
# nn.Module을 상속받아야 PyTorch 모델로 사용할 수 있다.
class MyLeNet5_2(nn.Module):

    # __init__ 함수는 모델에 사용할 layer들을 미리 정의하는 부분이다.
    def __init__(self):

        # 부모 클래스인 nn.Module의 초기화 함수를 호출한다.
        # PyTorch 모델을 만들 때 반드시 필요하다.
        super(MyLeNet5_2, self).__init__()


        # ==============================
        # Convolution Feature Extractor
        # ==============================

        # nn.Sequential은 여러 layer를 순서대로 묶어주는 기능이다.
        # 안에 넣은 layer들이 위에서 아래 순서대로 자동 실행된다.
        #
        # 즉, forward 함수에서
        # self.conv_1(x)
        # self.maxpool_1(x)
        # self.relu(x)
        # 이런 식으로 하나씩 쓰지 않아도 된다.
        self.conv_layers = nn.Sequential(

            # 첫 번째 convolution layer이다.
            #
            # 입력 channel 수는 1이다.
            # MNIST 같은 흑백 이미지는 channel이 1개이기 때문이다.
            #
            # 출력 channel 수는 6이다.
            # 즉, 6개의 filter를 사용한다는 뜻이다.
            #
            # kernel_size=5는 5x5 filter를 사용한다는 의미이다.
            #
            # padding=2는 입력 이미지의 테두리에 0을 2칸씩 추가한다는 뜻이다.
            # 입력이 28x28이고 kernel이 5x5일 때 padding=2를 주면
            # 출력 크기가 다시 28x28로 유지된다.
            nn.Conv2d(1, 6, kernel_size=5, padding=2),

            # 첫 번째 max pooling layer이다.
            #
            # nn.MaxPool2d(2)는 kernel_size=2라는 뜻이다.
            # stride를 따로 지정하지 않으면 stride도 2로 설정된다.
            #
            # 따라서 2x2 영역에서 가장 큰 값을 하나 고르고,
            # 두 칸씩 이동하면서 pooling을 수행한다.
            #
            # 결과적으로 가로세로 크기가 절반으로 줄어든다.
            # 28x28 -> 14x14
            nn.MaxPool2d(2),

            # ReLU activation function이다.
            #
            # 음수 값은 0으로 만들고,
            # 양수 값은 그대로 통과시킨다.
            #
            # 신경망에 비선형성을 추가하기 위해 사용한다.
            nn.ReLU(),


            # 두 번째 convolution layer이다.
            #
            # 입력 channel 수는 6이다.
            # 앞의 Conv2d layer가 6개의 feature map을 출력했기 때문이다.
            #
            # 출력 channel 수는 16이다.
            # 즉, 16개의 filter를 사용한다.
            #
            # kernel_size=5이므로 5x5 filter를 사용한다.
            #
            # padding을 지정하지 않았으므로 기본값은 padding=0이다.
            nn.Conv2d(6, 16, kernel_size=5),

            # Dropout layer이다.
            #
            # Dropout은 학습 중에 일부 뉴런의 출력을 랜덤하게 0으로 만든다.
            # 이렇게 하면 특정 뉴런에만 의존하는 것을 줄이고,
            # overfitting을 방지하는 데 도움을 준다.
            #
            # nn.Dropout()의 기본 dropout 확률은 p=0.5이다.
            # 즉, 학습 중에 약 50%의 값을 랜덤하게 0으로 만든다.
            #
            # 단, model.eval() 상태에서는 dropout이 적용되지 않는다.
            nn.Dropout(),

            # 두 번째 max pooling layer이다.
            #
            # 2x2 영역에서 가장 큰 값을 선택한다.
            # stride도 기본적으로 2이므로 가로세로 크기가 절반으로 줄어든다.
            nn.MaxPool2d(2),

            # 두 번째 ReLU activation function이다.
            #
            # Conv2d와 Pooling을 거친 결과에 비선형성을 추가한다.
            nn.ReLU(),


            # 세 번째 convolution layer이다.
            #
            # 입력 channel 수는 16이다.
            # 앞의 Conv2d layer가 16개의 feature map을 출력했기 때문이다.
            #
            # 출력 channel 수는 120이다.
            # 즉, 120개의 filter를 사용한다.
            #
            # kernel_size=5이므로 5x5 filter를 사용한다.
            #
            # 이 layer에 들어오기 전 feature map 크기는 16 x 5 x 5가 된다.
            # 여기에 5x5 kernel을 적용하면 출력 크기는 120 x 1 x 1이 된다.
            nn.Conv2d(16, 120, kernel_size=5)
        )


        # ==============================
        # Fully Connected Classifier
        # ==============================

        # fully connected layer들도 nn.Sequential로 묶는다.
        #
        # convolution layer에서 특징을 뽑은 뒤,
        # 이 부분에서 최종 class를 분류한다.
        self.fc_layers = nn.Sequential(

            # 첫 번째 fully connected layer이다.
            #
            # conv_layers의 마지막 출력은 120 x 1 x 1이다.
            # 이것을 flatten하면 120개의 값이 된다.
            #
            # 따라서 입력 크기는 120이고,
            # 출력 크기는 84이다.
            nn.Linear(120, 84),

            # fully connected layer 뒤에 ReLU를 적용한다.
            #
            # 음수 값은 0으로 만들고,
            # 양수 값은 그대로 통과시킨다.
            nn.ReLU(),

            # 두 번째 fully connected layer이다.
            #
            # 입력 크기는 84이다.
            # 출력 크기는 10이다.
            #
            # 출력 10개는 보통 MNIST 숫자 분류에서
            # 0부터 9까지 각 class에 대한 점수라고 볼 수 있다.
            nn.Linear(84, 10)
        )


    # forward 함수는 입력 데이터가 실제로 어떤 순서로 지나가는지 정의한다.
    def forward(self, x):

        # 입력 이미지를 convolution layer 묶음에 넣는다.
        #
        # self.conv_layers 안에는 Conv2d, MaxPool2d, ReLU, Dropout 등이
        # 순서대로 들어 있기 때문에 자동으로 차례대로 실행된다.
        #
        # 입력 크기 예시:
        # batch_size x 1 x 28 x 28
        #
        # 출력 크기 예시:
        # batch_size x 120 x 1 x 1
        x = self.conv_layers(x)

        # fully connected layer에 넣기 위해 flatten을 수행한다.
        #
        # 현재 x의 크기는 batch_size x 120 x 1 x 1이다.
        # 이것을 batch_size x 120 형태로 바꾼다.
        #
        # -1은 batch_size를 자동으로 맞추라는 뜻이다.
        # 120은 한 이미지당 feature 개수이다.
        x = x.view(-1, 120)  # flatten

        # flatten된 120개의 feature를 fully connected layer에 넣는다.
        #
        # self.fc_layers 안에는 Linear, ReLU, Linear가 순서대로 들어 있다.
        #
        # 최종 출력 크기:
        # batch_size x 10
        x = self.fc_layers(x)

        # 최종 결과를 반환한다.
        #
        # 여기서는 softmax를 따로 적용하지 않는다.
        # PyTorch에서 nn.CrossEntropyLoss를 사용할 경우,
        # loss 함수 내부에서 softmax 처리를 함께 하기 때문이다.
        return x


# 모델 객체를 생성한다.
model = MyLeNet5_2()

# 모델 구조를 출력한다.
# Sequential 안에 어떤 layer들이 들어 있는지 확인할 수 있다.
print(model)


# ==============================
# 동작 확인용 테스트 코드
# ==============================

# 임의의 입력 이미지를 만든다.
#
# 입력 형태는 다음 순서이다.
# batch_size, channel, height, width
#
# 여기서는 batch_size=1,
# channel=1,
# height=28,
# width=28인 흑백 이미지 한 장을 만든다.
dummy_input = torch.randn(1, 1, 28, 28)

# 모델에 dummy_input을 넣어서 결과를 확인한다.
dummy_output = model(dummy_input)

# 출력 크기를 확인한다.
# class가 10개이므로 torch.Size([1, 10])이 나온다.
print(dummy_output.shape)
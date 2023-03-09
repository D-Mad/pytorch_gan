import numpy as np

# Khởi tạo trọng số ngẫu nhiên cho mạng nơ-ron
W = np.random.randn(2, 1)

# Khởi tạo đầu vào và đầu ra thực tế
X = np.array([[1, 2], [3, 4], [5, 6]])
y_true = np.array([[2], [4], [6]])

# Định nghĩa hàm loss
def loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Định nghĩa hàm tính đạo hàm theo trọng số
def gradient(X, y_pred, y_true):
    return np.dot(X.T, (y_pred - y_true))

# Tốc độ học cố định
learning_rate = 0.01

# Huấn luyện mạng
for i in range(1000):
    # Truyền thẳng đầu vào qua mạng
    y_pred = np.dot(X, W)
    
    # Tính độ lỗi
    error = loss(y_pred, y_true)
    
    # Tính đạo hàm theo trọng số
    grad = gradient(X, y_pred, y_true)
    
    # Cập nhật trọng số bằng gradient descent
    W -= learning_rate * grad
    
    # In ra giá trị độ lỗi trong quá trình huấn luyện
    if i % 100 == 0:
        print("Epoch %d - Loss: %.4f" % (i, error))
        
# In ra trọng số sau khi huấn luyện
print("Trained weights:", W)
print("y^ la :",y_pred)

# Huấn Luyện Mô Hình GPT với PyTorch

Dự án này triển khai mô hình GPT (Generative Pre-trained Transformer) từ đầu bằng cách sử dụng PyTorch. Mã nguồn bao gồm định nghĩa kiến trúc mô hình trong `model.py` và kịch bản huấn luyện cùng sinh văn bản trong `train.py`. Dự án được thiết kế để huấn luyện mô hình ngôn ngữ trên dữ liệu văn bản, với mục tiêu hỗ trợ tác vụ dịch Anh-Việt (với dữ liệu nằm trong thư mục `data/train-en-vi`).

## Nội Dung Dự Án
```
.
├── model.py         # Định nghĩa kiến trúc mô hình GPT và các thành phần của Transformer
├── train.py         # Script huấn luyện, xử lý dữ liệu, tối ưu hóa, và sinh mẫu văn bản
├── data/
│   └── train-en-vi  # Thư mục chứa dữ liệu huấn luyện (dữ liệu nhị phân và file meta.pkl)
└── out/             # Thư mục lưu checkpoint của mô hình sau huấn luyện
```
- **model.py**  
  - **LayerNorm**: Lớp chuẩn hóa layer, hỗ trợ tùy chọn sử dụng bias.
  - **CausalSelfAttention**: Cơ chế self-attention có tính chất causal, đảm bảo mỗi token chỉ “chú ý” đến các token trước đó.
  - **MLP**: Mạng đa lớp (feed-forward network) dùng sau khối attention.
  - **Block**: Một khối Transformer kết hợp giữa attention và MLP cùng với normalization.
  - **GPTConfig & GPT**: Cấu hình và định nghĩa mô hình GPT, bao gồm embedding, các khối Transformer, và đầu ra dự đoán token.

- **train.py**  
  - Tải và xử lý dữ liệu từ file nhị phân trong thư mục `data/train-en-vi`.
  - Cấu hình tham số huấn luyện (batch size, block size, số layer, số head, kích thước embedding, v.v.).
  - Thiết lập mô hình, bộ tối ưu (AdamW, hỗ trợ fused nếu có), và chế độ huấn luyện với mixed precision.
  - Vòng lặp huấn luyện với logging loss, điều chỉnh learning rate theo lịch trình warmup & decay cosine, và gradient clipping.
  - Sinh mẫu văn bản định kỳ thông qua hàm `generate`.
  - Lưu checkpoint mô hình cuối cùng vào thư mục `out`.

## Yêu Cầu & Cài Đặt

- **Yêu cầu hệ thống:**  
  - Python 3.7 trở lên  
  - PyTorch (với CUDA nếu sử dụng GPU)  
  - NumPy  
  - tiktoken (dùng để mã hóa văn bản theo chuẩn GPT-2)

- **Cài đặt:**  
  Cài đặt các gói cần thiết bằng pip:
  ```bash
  pip install torch numpy tiktoken

## Cách Sử Dụng

1. **Chuẩn Bị Dữ Liệu**

Khi huấn luyện GPT‑2, mô hình được cung cấp các khối đầu vào có kích thước cố định (ví dụ: 1024 token). Thông thường, bạn không có mỗi ví dụ huấn luyện dài đúng 1024 token – thay vào đó, bạn tạo ra một luồng dài văn bản đã được mã hóa thành token và sau đó chia nó thành các khối liền mạch chứa 1024 token.

Ví dụ, nếu file huấn luyện kết hợp của bạn trông như sau:
```
EN: I love machine learning.
VI: Tôi yêu học máy.
<|endoftext|>
EN: How are you today?
VI: Bạn có khỏe không hôm nay?
<|endoftext|>
EN: The weather is nice and sunny.
VI: Thời tiết đẹp và nắng.
<|endoftext|>
...
```
Khi bạn mã hóa file này thành token, bạn nhận được một danh sách dài các ID token. Trong quá trình tiền xử lý, danh sách này được chia thành các đoạn gồm 1024 token. Một đoạn như vậy có thể bắt đầu từ giữa một ví dụ và kết thúc giữa một ví dụ khác.

Ví dụ, một khối đầu vào (giả định) gồm 1024 token có thể trông như sau khi giải mã lại thành văn bản (lưu ý rằng quá trình mã hóa và giải mã token có thể không khôi phục hoàn toàn định dạng ban đầu):
```
EN: I love machine learning.
VI: Tôi yêu học máy.
<|endoftext|>
EN: How are you today?
VI: Bạn có khỏe không hôm nay?
<|endoftext|>
EN: The weather is nice and sunny.
VI: Thời tiết đẹp và nắng.
<|endoftext|>
EN: What time is it?
VI: Mấy giờ rồi?
<|endoftext|>
...
```
Khối này có thể bao gồm nhiều ví dụ đầy đủ, cùng với một phần của ví dụ bổ sung nếu ranh giới 1024 token rơi vào giữa ví dụ.

Tóm lại, mặc dù dữ liệu huấn luyện của bạn gồm nhiều ví dụ song ngữ riêng lẻ, nhưng chúng được nối lại thành một luồng văn bản dài và sau đó được chia thành các khối có 1024 token. Điều này đảm bảo rằng mỗi đầu vào huấn luyện cho mô hình có kích thước cố định như yêu cầu của GPT‑2.

2. **Huấn Luyện Mô Hình**
  - Chạy kịch bản huấn luyện:
  ```
  python train.py
  ```
  Các tham số huấn luyện như số layer, số head, kích thước embedding, learning rate, v.v. có thể được điều chỉnh trực tiếp trong file train.py.

3. **Sinh Văn Bản**
  - Trong quá trình huấn luyện, sau mỗi khoảng thời gian nhất định (ví dụ: mỗi 2000 iterations), mô hình sẽ sinh ra văn bản mẫu dựa trên prompt ban đầu.
  - Kết quả sinh mẫu sẽ được in ra console.

4. **Lưu & Tải Mô Hình**
   
- Sau khi huấn luyện xong, mô hình cùng trạng thái của bộ tối ưu được lưu vào file out/ckpt.pt.
- Bạn có thể tải lại checkpoint này để tiếp tục huấn luyện hoặc thực hiện sinh văn bản.

## Liên Hệ & Hỗ Trợ
Nếu bạn có bất kỳ câu hỏi hoặc gặp vấn đề gì trong quá trình sử dụng, vui lòng mở issue trên repository hoặc liên hệ với tác giả để được hỗ trợ.



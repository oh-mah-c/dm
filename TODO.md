Đúng, mình dừng phần sửa tiếp và liệt kê các file có dấu hiệu **draft / simplified / omitted / hack / không chuẩn paper** để bạn tải paper rồi implement lại.

**Cần Ưu Tiên Cao**
Các file này có comment rất rõ là chưa chuẩn paper:

- `src/algorithms/chui_miner.c`
  - Có `simplified here`, `For now, this baseline...`
  - CHUI-Miner đang không nên tin là paper-accurate.

- `src/algorithms/efim_closed.c`
  - Có `hack`, `not implemented in this draft`, `Logic omitted`
  - EFIM-Closed chắc chắn là draft.

- `src/algorithms/chuimine.c`
  - Có `Actually, CHUI-Mine uses a specific set management`, `Simplified`
  - CHUI-Mine Closed/Maximal cần implement lại.

- `src/algorithms/fhim.c`
  - Có `Simplified`, `But for 100% fidelity...`
  - FHIM trong paper HUCI cũng cần implement lại chuẩn nếu muốn claim chính xác.

- `src/algorithms/huim_su.c`
  - Có `simplified`, `FALLBACK`, `Implementation omitted`
  - Không nên dùng cho paper claim.

- `src/algorithms/dfi_growth.c`
  - Có `Simple miner for now`, `simplified`, `omitted`
  - Cần tải paper/đọc lại nếu dùng.

- `src/algorithms/up_growth.c`
  - Có `For the sake...`, `Actually, I'll just...`, `Simplified`
  - UP-Growth chưa đáng tin.

**Cần Rà Lại**
Có dấu hiệu không hoàn toàn paper-exact hoặc dùng approximation/basic logic:

- `src/algorithms/ihup.c`
- `src/algorithms/close.c`
- `src/algorithms/finplus.c`
- `src/algorithms/slim.c`
- `src/algorithms/thui.c`
- `src/algorithms/feacp.c`
- `src/algorithms/apriori_hybrid.c`
- `src/algorithms/cfi_stream.c`
- `src/algorithms/eihi.c`
- `src/algorithms/fhm.c`
- `src/algorithms/krimp.c`

**Riêng HUCI**
- `src/algorithms/huci_miner.c` hiện đã khớp paper example: `20` HUI, `10` HUCI, `8` HGB rules.
- Nhưng vì nó dùng HUI enumeration riêng, nếu bạn muốn Q1-level thì vẫn nên tải/đọc kỹ paper HUCI + FHIM và implement lại FHIM chuẩn trong cùng pipeline, thay vì dựa vào phần mình vừa sửa nhanh.

**Kết luận ngắn**
Các thuật toán closed-HUI đang đáng nghi nhất: `chui_miner`, `efim_closed`, `chuimine`, `cls_miner` cần paper gốc và implement lại sạch. Lý do count lệch là vì ít nhất vài implementation hiện tại không phải bản paper-accurate, mà là draft/simplified.
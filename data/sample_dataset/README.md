# Hệ Thống Bán Lẻ Gia Dụng & Tiện Ích RetailVN
as_of_date: 2024-06-15

## Mô tả dữ liệu
1. orders.csv: Dữ liệu giao dịch đơn hàng bán lẻ.
   - order_id: Mã định danh đơn hàng (integer)
   - created_at: Thời gian tạo đơn hàng (YYYY-MM-DD HH:MM:SS)
   - customer_id: Mã khách hàng (string)
   - branch: Chi nhánh thực hiện (Hà Nội, TP HCM, Đà Nẵng)
   - amount: Tổng giá trị đơn hàng (VNĐ)
   - status: Trạng thái đơn ('Thành công', 'Đang xử lý', 'Đã hủy')

2. products.xlsx: Danh mục hàng hóa và kho hàng.
   - Sheet 'Products': product_id, product_name, category, price.
   - Sheet 'Inventory': product_id, stock_quantity, warehouse.

3. database.sqlite: Cơ sở dữ liệu thông tin khách hàng.
   - Bảng 'customers': customer_id, customer_name, city, join_date.

# Predictive Analytics - Mohammad Radya Fariez
--------------------------------------------------
Disusun oleh : **Mohammad Radya Fariez**

Berikut merupakan laporan submission _project_ pertama pada kelas Dicoding (Machine Learning Terapan)

**Domain Proyek**
--------------------------------------------------
**Latar Belakang**

Kendaraan roda empat merupakan salah satu alat transportasi yang diandalkan pada masa modern ini. Seiring berkembangnya zaman, kendaraan beroda empat lahir dengan beberapa kegunaan, salah satunya penggunaan sebagai mobil pribadi. Mobil pribadi di beberapa daerah tertentu merupakan alat yang diandalkan terutama di wilayah yang minim transportasi umum. Penggunaannya yang praktis membuat sebagian masyarakat menggunakan kendaraan mobil, sehingga permintaan/_demand_ pembelian mobil terus meningkat signifikan sejak era industri modern hingga sekarang.

![New-santa-fe-2021](https://user-images.githubusercontent.com/109395960/204444807-4d9fd0bb-f2b0-47e6-b77a-945e3cf42097.jpg)

Dengan banyaknya minat masyarakat terhadap pemakaian kendaraan mobil, maka opsi penyediaannya juga beragam. Salah satu opsi pembelian mobil dan kendaraan pribadi lainnya yang menjadi pilihan sebagian masyarakat adalah pembelian kendaraan _secondhand_/bekas, karena harganya yang lebih terjangkau jika dibandingkan dengan harga baru. Maka, dengan adanya opsi tersebut masyarakat bebas memilih jenis mobil yang diinginkan berdasarkan spesifikasi dan umur pada mobil.

Maka dari itu, Machine Learning dapat dimanfaatkan sebagai alat bantu untuk melakukan analisa terhadap pemilihan produk mobil yang tepat bagi calon pembeli mobil. Calon pembeli mobil hanya perlu memberikan input terhadap spesifikasi mobil yang diinginkan, sehingga Machine Learning dapat memberikan _output_ harga dari mobil berdasarkan dataset yang diolah sebelumnya. 

**Business Understanding**
--------------------------------------------------
Proyek ini dapat diimplementasikan pada perusahaan dengan karakteristik sebagai berikut:

- Perusahaan penjual mobil bekas yang memberi opsi beberapa mobil ke calon konsumen
- Perusahaan penyedia atau jual beli kendaraan secara _online_ yang dapat memberikan fitur spesifikasi pada pengguna aplikasi.

**Problem Statement**
--------------------------------------------------
- Spesifikasi apa yang paling berpengaruh terhadap tinggi/rendahnya harga mobil?
- Berapa harga pasar mobil _second_ dengan spesifikasi tertentu?
- Bagaimana cara melakukan proses data agar model dapat memberikan _output_ data yang baik?

**Goals**
--------------------------------------------------
- Mengetahui spesifikasi yang paling berpengaruh terhadap tinggi/rendahnya harga mobil
- Mengetahui harga pasar mobil _second_ spesifikasi tertentu dengan melakukan _preprocessing_ pada model
- Membuat model Machine Learning untuk memprediksi harga mobil dengan spesifikasi yang diinginkan.

**Solution Statement**
--------------------------------------------------
- Melakukan Univariate Analysis dan Multivariate Analysis serta melakukan visualisasi data sehingga dapat mengetahui korelasi antar fitur
- Mempersiapkan data agar dapat diproses untuk membangun model
- Membangun tiga model, yaitu: K-Nearest Neighbor, Random Forest dan Boosting Algorithm untuk membandingkan hasil yang paling optimal.

**Data Understanding**
--------------------------------------------------
Dataset yang digunakan pada proyek ini bersumber dari Kaggle yang dapat diunduh di [Vehicle Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

Dari dataset terkait, didapatkan informasi sebagai berikut:
- Dataset merupakan data yang memiliki format CSV (Comma-Separated-Values)
- Dataset memiliki 4339 sampel data
- Dataset memiliki 7 fitur

Beberapa variable pada dataset, yaitu:
- **name**: Nama merk mobil
- **year** : Tahun produksi pada mobil
- **selling_price**: Harga jual
- **km_driven**: Jumlah jarak tempuh pada mobil (dalam kilometer)
- **fuel**: Jenis bahan bakar
- **seller_type**: Jenis penjualan
- **transmission**: Jenis transmisi mobil
- **owner**: Jenis kepemilikan sebelumnya.

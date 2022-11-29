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

Pada variabel - variabel tersebut diketahui bahwa variabel **name** tidak mempengaruhi harga mobil, sehingga fitur tersebut dapat dihapus.

**Univariate Analysis**
--------------------------------------------------
Pada tahap ini dilakukan analisa fitur secara terpisah.

Berikut analisa terhadap fitur kategorik:

Keempat fitur kategorik, diantaranya, fitur **seller**, **transmission**, **fuel**, dan **owner** memiliki sebaran data yang tidak terlalu merata, namun bervariasi.

<img width="120" alt="fitur fuel" src="https://user-images.githubusercontent.com/109395960/204485938-939e0ade-b003-4e0c-80ac-ebe7278ea1f6.PNG">
<img width="170" alt="fitur owner" src="https://user-images.githubusercontent.com/109395960/204485942-d9c1af24-0298-4c47-9c34-54198878a0fe.PNG">
<img width="156" alt="fitur seller type" src="https://user-images.githubusercontent.com/109395960/204485945-093b1026-084e-4042-82cb-699b370ba1a7.PNG">
<img width="161" alt="fitur transmission" src="https://user-images.githubusercontent.com/109395960/204485949-fed73e36-1c9c-45d4-94d2-a66719775abe.PNG">

Lalu, pada fitur numerik, dapat ditunjukkan analisa sebaran dalam bentuk grafik:

<img width="583" alt="fitur numerik1" src="https://user-images.githubusercontent.com/109395960/204486781-0d12b15e-092e-496b-8db4-ff2b9e297c28.PNG">
<img width="268" alt="fitur numerik2" src="https://user-images.githubusercontent.com/109395960/204486797-831a615c-be54-4fd9-81d1-fd27decb9783.PNG">

Pada sebaran data yang ditunjukkan, dapat diketahui bahwa:

- Tahun produksi mobil yang paling banyak dijual yaitu mobil dengan produksi tahun 2017
- Jarak tempuh yang dimiliki mobil paling banyak berada dibawah rentang 100.000 kilometer.

**Multivariate Analysis**
--------------------------------------------------
**Analisa Fitur**

Pada bagian ini dibuat Multivariate Analysis, yaitu korelasi antara 2 atau lebih fitur dalam data.

Dibawah ini didapatkan Correlation Matrix

<img width="289" alt="correlation matrix" src="https://user-images.githubusercontent.com/109395960/204488914-250c6ce9-86a5-43ac-8020-f21806992a31.PNG">

Pada Correlation Matrix ditunjukkan bahwa ketiga fitur memiliki korelasi yang merata secara signifikan.

**Data Preparation**








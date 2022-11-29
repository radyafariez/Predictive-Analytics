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
--------------------------------------------------
Dilakukan beberapa step pada proses Data Preparation, diantaranya:

- **One Hot Encoding**

Merupakan teknik dalam mengubah fitur kategorik menjadi numerik. Dalam proyek ini, dilakukan proses One Hot Encoding pada fitur **fuel**, **seller_type**, **transmission**, dan **owner**.

- **Normalization**

Merupakan proses penyeragaman pada data tiap fitur dan label, sehingga Machine Learning dapat melakukan proses data yang memiliki skala relatif sama. Di dalam proyek ini dilakukan Normalisasi dengan teknik StandardScaler.

- **Train Test Split**

Merupakan proses atau step pembagian dataset menjadi data latih dan data test. Di dalam Machine Learning perlu dilakukan pembagian kedua data. Dalam hal ini, data test diberi proporsi 0.1 atau 10% dari keseluruhan pada Dataset.

**Modelling**
--------------------------------------------------
Pada proyek ini dilakukan deploy pada 3 algoritma, yaitu KNN, Random Forest dan Boost Algorithm.

- **KNN** atau K-Nearest Neighbor merupakan algortima dengan membandingkan jarak sampel dengan sampel lain dengan melihat jumlah n data terdekat. Dalam proyek ini, digunakan KNeighborsRegressor dengan memberikan input X_train dan y_train.
- **Random Forest** merupakan teknik yang  dilakukan dengan metode Ensemble. Dilakukan pengoperasian dengan membangun decision tree (pohon keputusan). Pada proyek ini digunakan RandomForestRegressor.
- **Adaptive Boosting** Merupakan algoritma yang dijalankan untuk meningkatkan performa dengan cara melakukan merging atau penggabungan model yang dianggap lemah dan digantikannya dengan model yang kuat. Dalam proyek ini, digunakan AdaBoostRegressor.

**Evaluation**
--------------------------------------------------
Dalam step ini, evaluasi dilakukan dengan memperhitungkan Mean Squared Error yang dapat menentukan ttingkat kemiripan antara hasil prediksi dengan nilai y_test (hasil nyata). dalam MSE ini, tingkatan error dihitung berdasarkan rata - rata error dari kuadrat hasil aktual yang diselisihkan hasil prediksi.

Maka, dari perhitungan tersebut, dihasilkan beberapa evaluasi sebagai berikut:

- MSE

<img width="245" alt="mse" src="https://user-images.githubusercontent.com/109395960/204513100-d4e620e1-c391-4124-8d49-b1052dc276ec.PNG">

- Nilai Akurasi

<img width="409" alt="graph" src="https://user-images.githubusercontent.com/109395960/204513090-a4cf5a50-b371-4996-a5ad-0af9b5804b73.PNG">

- Nilai Perbandingan

<img width="294" alt="comparison" src="https://user-images.githubusercontent.com/109395960/204513105-34f2b309-ed8c-4339-973b-db8260960502.PNG">

Dari beberapa hasil model, dapat diketahui bahwa algoritma Random Forest memiliki nilai akurasi yang lebih baik dengan tingkatan error terkecil. Nilai perbandingan juga menunjukkan bahwa Random Forest Algorithm memiliki nilai yang paling mendekati y_true (nilai aktual) dibanding model algoritma lainnya.

# Predictive Analytics - Mohammad Radya Fariez
--------------------------------------------------
Disusun oleh : **Mohammad Radya Fariez**

Berikut merupakan laporan submission _project_ pertama pada kelas Dicoding (Machine Learning Terapan)

**Domain Proyek**
--------------------------------------------------
**Latar Belakang**

Kendaraan roda empat merupakan salah satu alat transportasi yang diandalkan pada masa modern ini. Seiring berkembangnya zaman, kendaraan beroda empat lahir dengan beberapa kegunaan, salah satunya penggunaan sebagai mobil pribadi. Mobil pribadi di beberapa daerah tertentu merupakan alat yang diandalkan terutama di wilayah yang minim transportasi umum. Penggunaannya yang praktis membuat sebagian masyarakat menggunakan kendaraan mobil, sehingga permintaan/_demand_ pembelian mobil terus meningkat signifikan sejak era industri modern hingga sekarang.

![New-santa-fe-2021](https://user-images.githubusercontent.com/109395960/204444807-4d9fd0bb-f2b0-47e6-b77a-945e3cf42097.jpg)

Dengan banyaknya minat masyarakat terhadap pemakaian kendaraan mobil, maka opsi penyediaannya juga beragam. Salah satu opsi pembelian mobil dan kendaraan pribadi lainnya yang menjadi pilihan sebagian masyarakat adalah pembelian _used car_/mobil bekas, karena harganya yang lebih terjangkau jika dibandingkan dengan harga baru. Maka, dengan adanya opsi tersebut masyarakat bebas memilih jenis mobil yang diinginkan berdasarkan spesifikasi dan umur pada mobil.

Maka dari itu, Machine Learning dapat dimanfaatkan sebagai alat bantu untuk melakukan analisa terhadap pemilihan produk mobil yang tepat bagi calon pembeli mobil. Calon pembeli mobil hanya perlu memberikan input terhadap spesifikasi mobil yang diinginkan, sehingga Machine Learning dapat memberikan _output_ harga dari mobil berdasarkan dataset yang diolah sebelumnya. 

**Business Understanding**
--------------------------------------------------
Proyek ini dapat diimplementasikan pada perusahaan dengan karakteristik sebagai berikut:

- Perusahaan penjual mobil bekas yang memberi opsi beberapa mobil ke calon konsumen
- Perusahaan penyedia atau jual beli kendaraan secara _online_ yang dapat memberikan fitur spesifikasi pada pengguna aplikasi.

**Problem Statement**
--------------------------------------------------
- Spesifikasi apa yang paling berpengaruh terhadap tinggi/rendahnya harga mobil?
- Berapa harga pasar mobil bekas dengan spesifikasi tertentu?
- Bagaimana cara melakukan proses data agar model dapat memberikan _output_ data yang baik?

**Goals**
--------------------------------------------------
- Mengetahui spesifikasi yang paling berpengaruh terhadap tinggi/rendahnya harga mobil
- Mengetahui harga pasar mobil bekas spesifikasi tertentu dengan melakukan _preprocessing_ pada model
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
- Tidak terdapat _missing value_ di dalam dataset
- Dataset memiliki 4339 sampel data
- Dataset memiliki 7 fitur
- Dari 7 fitur yang tersedia, terdapat 4 fitur dengan tipe int64, 3 tipe lainnya bertipe object

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

| fuel     |               |
|----------|---------------|
| CNG      |  	   40      |
| Diesel   |     2153      |
| Electric |        1      |
| LPG      |       23      |
| Petrol   |     2123      |
| Name: fuel, dtype: int64


| owner      	       |           |
|--------------------|-----------|
| First Owner	       |   2832    |
| Fourth & AboveOwner|     81    |
| Second Owner	     |   1106    |
| Test Driver Car    |     17    |
| Third Owner	       | 	  306    |
| Name: owner, dtype: int64


| seller_type        |           |
|--------------------|-----------|
| Dealer	           |    994    |
| Individual	       |   3244    |
| Trustmark Dealer   |    102    |
| Name: seller_type, dtype: int64


| transmission       |           |
|--------------------|-----------|
| Automatic	         |    448    |
| Manual|   3244     |	  3892	 |
| Name: seller_type, dtype: int64

Lalu, pada fitur numerik, dapat ditunjukkan analisa sebaran dalam bentuk grafik:

<img width="583" alt="fitur numerik1" src="https://user-images.githubusercontent.com/109395960/204486781-0d12b15e-092e-496b-8db4-ff2b9e297c28.PNG">
<img width="268" alt="fitur numerik2" src="https://user-images.githubusercontent.com/109395960/204486797-831a615c-be54-4fd9-81d1-fd27decb9783.PNG">

Pada sebaran data yang ditunjukkan, dapat diketahui bahwa:

- Tahun produksi mobil yang paling banyak dijual yaitu mobil dengan produksi tahun 2017
- Jarak tempuh yang dimiliki mobil paling banyak berada dibawah rentang 100.000 kilometer.

**Multivariate Analysis**
--------------------------------------------------
**Analisa Fitur Numerik**

Pada bagian ini dibuat Multivariate Analysis, yaitu korelasi antara 2 atau lebih fitur dalam data.

|      | year 	      | selling_price  | km_driven     |
|------|--------------|----------------|---------------|
| count| 4340.000000  |	 4.340000e+03  | 4340.000000   |
| mean | 2013.090783  |	 5.041273e+05  | 66215.777419  |
| std  | 4.215344     |	 5.785487e+05  | 46644.102194  |
| min  | 1992.000000  |	 2.000000e+04  | 1.000000      |
| 25%  | 2011.000000  |	 2.087498e+05  | 35000.000000  |
| 50%  | 2014.000000  |	 3.500000e+05  | 60000.000000  |
| 75%  | 2016.000000  |	 6.000000e+05  | 90000.000000  |
| max  | 2020.000000  |	 8.900000e+06  | 806599.000000 |

Pada tabel diatas, terdapat data statistik pada fitur numerik. Dari tabel tersebut diketahui nilai _mean_, nilai minimum, tengah dan maksimum tiap masing - masing fitur untuk lebih memahami karakteristik data.

Dibawah ini didapatkan Correlation Matrix

<img width="289" alt="correlation matrix" src="https://user-images.githubusercontent.com/109395960/204488914-250c6ce9-86a5-43ac-8020-f21806992a31.PNG">

Pada Correlation Matrix ditunjukkan bahwa ketiga fitur memiliki korelasi yang merata secara signifikan.

**Analisa Fitur Kategorik**

Fitur kategorik pada proyek ini diantaranya fitur **transmission**, **owner**, **seller_type**, dan **fuel**. Pada grafik dibawah ini ditunjukkan grafik _bar_ representasi korelasi antara taget (selling_price) dengan fitur kategorik sebagai berikut:

- Fitur Fuel

<img width="580" alt="selling price x fuel" src="https://user-images.githubusercontent.com/109395960/204957136-13d37588-d9be-48fb-834a-e400a80c5a8b.PNG">

Dari grafik pada gambar atas dapat diketahui mobil berjenis bahan bakar Diesel cenderung mempengaruhi tingginya harga mobil, sedangkan jenis bahan bakar LPG cenderung memiliki harga mobil yang paling rendah. Pada fitur ini, cukup memberi pengaruh yang besar terhadap tinggi rendahnya harga.

- Fitur Seller Type

<img width="592" alt="seller type x selling price" src="https://user-images.githubusercontent.com/109395960/204957904-ecc9839d-617f-4093-840a-4c726b1ae0b5.PNG">

Pada fitur Seller Type, dapat diketahui bahwa Trustmark Dealer memiliki harga yang lebih tinggi pada mobil dibanding jenis tempat penjualan mobil lainnya pada fitur ini.

- Fitur Transmission

<img width="579" alt="transmission x selling price" src="https://user-images.githubusercontent.com/109395960/204958147-2815a20c-94f6-4e26-bf49-855762225ce6.PNG">

Pada fitur ini, hanya ada 2 jenis transmisi mobil, akan tetapi harga mobil dengan jenis transmisi otomatis (Automatic) akan cenderung lebih mahal dibandingkan mobil transmisi manual.

- Fitur Owner

<img width="582" alt="owner x selling price" src="https://user-images.githubusercontent.com/109395960/204958339-395d3b49-24ad-4018-835a-f4bb9d450095.PNG">

Rata-rata harga jual relatif terhadap fitur 'owner' cukup beragam jenisnya. Dalam hal ini, mobil yang dijual dalam kondisi sebagai mobil _test drive_ atau uji coba cenderung lebih tinggi harganya, disusul jenis _First Owner_ atau pemilik pertama. Jenis tersebut lebih tinggi harganya jika dibandingkan jenis mobil yang kondisinya sebagai pemilik kedua, ketiga, dan seterusnya.

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

- **KNN** atau K-Nearest Neighbor merupakan algortima dengan membandingkan jarak sampel dengan sampel lain dengan melihat jumlah n data terdekat. Dalam proyek ini, digunakan KNeighborsRegressor dengan memberikan input X_train dan y_train. Pada algoritma ini, parameter yang digunakan, yaitu:

        n_neighbors = Jumlah K tetangga dengan jarak terdekat

- **Random Forest** merupakan teknik yang  dilakukan dengan metode Ensemble. Dilakukan pengoperasian dengan membangun decision tree (pohon keputusan). Pada proyek ini digunakan RandomForestRegressor. Adapun parameter yang digunakan pada model ini, diantaranya:

        random_state = Berperan untuk mengontrol _seed_ acak pada tiap iterasi boosting
        n_estimators = Jumlah maksimal estimator saat boosting dihentikan
        max_depth = Kedalaman maksimal pada pohon (tree).

- **Adaptive Boosting** Merupakan algoritma yang dijalankan untuk meningkatkan performa dengan cara melakukan merging atau penggabungan model yang dianggap lemah dan digantikannya dengan model yang kuat. Dalam proyek ini, digunakan AdaBoostRegressor. Berikut parameter yang digunakan pada algoritma ini:
        
        random_state = Berperan untuk mengontrol _seed_ acak pada tiap iterasi boosting
        n_estimators = Jumlah maksimal estimator saat boosting dihentikan
        learning_rate = Parameter yang digunakan untuk memperkuat kontribusi pada tiap regresor.
        
**Evaluation**
--------------------------------------------------
Dalam step ini, evaluasi dilakukan dengan memperhitungkan Mean Squared Error yang dapat menentukan ttingkat kemiripan antara hasil prediksi dengan nilai y_test (hasil nyata). dalam MSE ini, tingkatan error dihitung berdasarkan rata - rata error dari kuadrat hasil aktual yang diselisihkan hasil prediksi.

$$
\frac{1}{n} \sum_{i = 1}^{n} (Y_{i} - \hat{Y}_i)^2
$$

$MSE$ = Mean Squared Error

$n$		 = Number of data

$Y_{i}$ = Observed Values

$\hat{Y}_i$= Predicted Values

Maka, dari perhitungan tersebut, dihasilkan beberapa evaluasi sebagai berikut:

- MSE

| 	       |           train|            test|
|----------|----------------|----------------|
| KNN		   |90571408.006381 |249200473.92627 |		 
| RF	     |25508089.709651 |238771672.382782| 
| Boosting |129113213.176645|270278743.29991 |

- Nilai Akurasi

<img width="409" alt="graph" src="https://user-images.githubusercontent.com/109395960/204513090-a4cf5a50-b371-4996-a5ad-0af9b5804b73.PNG">

- Nilai Perbandingan

| 	  |  	y_true| prediksi_KNN| prediksi_RF| prediksi_Boosting|	
|-----|---------|-------------|------------|------------------|
| 527 |2750000	|2154166.7	  |2307842.9   | 1914371.6	      |

Dari beberapa hasil model, dapat diketahui bahwa algoritma Random Forest memiliki nilai akurasi yang lebih baik dengan tingkatan error terkecil. Nilai perbandingan juga menunjukkan bahwa Random Forest Algorithm memiliki nilai yang paling mendekati y_true (nilai aktual) dibanding model algoritma lainnya.

Porject 1
  2021-11-6 
    提交了task1-extratree_with_0.64_cross_validation.ipynb文件， 在一些时候，cross-validation score可以达到0.64
    文件修改了impute方法，先对数据进行median补值，然后去除outliers；然后再进行KNN impute, impute时，对X_test使用的是X_train的mean和std
    文件使用了lgb训练模型， 提高num_leaves，max_depth，num_iterations可让模型更加拟合

    下一步： 改进feature selection

  2021-11-7 晚 20：56 
   提交了task1-lgb-correlation_reduction
   增加了feature correlation reduction
   feature selection 量级在100左右
   validation score 能到0.648

   2021-11-8 晚 20：56 
   提交了GP-lkl
   在刘书记开创性GP的基础上增加了feature correlation reduction
   feature selection 54-50 pearson threshold 0.001, 0.96
   validation score 能到0.669


   2021-11-8 晚 22：48 
   提交了gp_lkl_model_merge
   在刘书记开创性GP和lgb的基础上，融合了lgb和gp， validation score能到0.69 开心！！！ ：-）

   2021-11-12 下午 16：01 
  提交了three_merge
  在组长和书记开创性的工作基础上， 融合了gp(rbf kernel)， gp(quadratic kernel)以及lgb， 
  validation score达到了 0.6945， 但是public sccore 提升不明显

   2021-11-12 下午 19：08 
  修改了three_merge中gp(rbf kernel)方法的feature选择的参数,同时用了shuffle
  validation score达到了 0.696， public sccore 提升至0.756
  
Project2
  2021-11-30 下午 15：30
  提交了 特征精选_基于8163_效果822以及特征海选_based_on_fish2。 先在fisht2（带fft）的基础上选一批可能的种子feature，
  然后在组长8163的模型上精选feature， 最后发现fft+wavelet+half of bruteforce feature效果最好

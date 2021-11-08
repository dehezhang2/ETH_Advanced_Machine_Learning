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

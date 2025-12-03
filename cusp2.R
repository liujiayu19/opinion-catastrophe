# 需要提前下载
install.packages("readr")
install.packages("lmtest")
install.packages("cusp")
install.packages("logistf")
install.packages("corrplot")

library(readr)
library(lmtest)
library(cusp)
library(logistf)

getwd()

# 读取数据
dada_normalized <- read.csv("D:/R_project/dada_variables_normalized.csv")
View(dada_normalized)
colnames(dada_normalized)



library(corrplot)
# 计算相关矩阵
correlation_matrix <- cor(dada_normalized)
# 绘制相关矩阵图
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", 
         addCoef.col = "black", # 添加相关系数
         tl.col = "black", tl.srt = 45) # 文字颜色和角度
# 通过相关矩阵可以看出，由于目前多重共线性的因素，一般来讲需要剔除几个变量





##### 不考虑上面问题后的尝试（后面可以看到，近乎完全相关的会自动被剔除）

# 提取出自变量和因变量
x_vars <- dada_normalized[, c("state_sentiment", "minus_sentiment", "comment_variance", 
                              "frequency", "topword_sentiment", "num_nodes", "num_edges", "density",
                              "avg_degree_centrality", "avg_betweenness_centrality", "avg_closeness_centrality")]
y_var <- dada_normalized[["follower_sentiment"]]
y_var_logi <- as.factor(ifelse(y_var>0,1,0))



# 普通线性回归（一般来讲这样操作，但cusp包中有直接对比模型的方法
# 且需要被对比的普通线性回归也是经过特殊设定的，因此这段内容可以忽略）
 linear_model <- lm(y_var ~ ., data = x_vars)
 linear_model_summary <- summary(linear_model)
# 
 linear_r_square <- linear_model_summary$r.squared
 linear_log_likelihood <- logLik(linear_model)
 linear_aic <- AIC(linear_model)
 linear_bic <- BIC(linear_model)


# cusp的简单信息：
# 在cusp模型中，alpha和beta分别代表"cusp"模型中的正常分布因子（asymmetry variable）
# 和分裂因子（bifurcation variable）。对于任何cusp模型，我们都需要指定这两个影响因子。

# 在cusp模型的公式表示中，y ~ f | v | u：
# y 是你的因变量（也就是你的响应变量）。
# f 是你的对称变量（也称为正常分布因子或者"asymmetry variable"），用于描述系统的稳定状态。
# 在一些场景中，f 也可以被解释为推动系统向特定方向发展的因素。
# 在cusp包的公式中，这个对称变量是alpha。
# v 是你的分裂变量（也称为 bifurcation variable），在一些场景中，v可以被理解为引发剧烈变化
# 或者引发系统状态从一个稳定状态切换到另一个稳定状态的因素。在cusp包的公式中，
# 这个分裂变量是beta。
# u 是一个可选项，代表在非线性回归分析中的随机误差变量。在大多数场景中，我们可以忽略u。

# 这些变量需要根据实际数据和研究背景进行选择。你可以基于你的理论知识和理解来选择哪些变量
# 作为alpha和beta，也可以基于你的实证数据进行选择。


# 在pdf文中，alpha和beta的选择很相似，仅各去除了一个变量
# 现在不严谨的选择（可以看做一个例子，根据具体对模型的看法来调整）：
# "follower_sentiment" 是因变量，表示回帖的平均情绪。

# "state_sentiment" 是对称变量（alpha），表示原始发帖的情绪。

# "minus_sentiment"，"comment_variance"，"frequency"，"num_nodes"，
# "density"，"avg_degree_centrality"，"avg_betweenness_centrality"，
# "avg_closeness_centrality" 是影响对称变量alpha的其他因素，可能影响社交媒体的整体情绪状态。

# "topword_sentiment" 和 "num_edges" 是分裂变量（beta），可能引发情绪状态的剧变。
# "topword_sentiment"表示主要词汇的情绪倾向，"num_edges"表示网络的连接数。

# Cusp回归
cusp_model <- cusp(follower_sentiment ~ state_sentiment ,
                   alpha ~ minus_sentiment + comment_variance + frequency + 
                     num_nodes + density + 
                     avg_degree_centrality + avg_betweenness_centrality + avg_closeness_centrality,
                   beta ~ topword_sentiment + num_edges,
                   data = dada_normalized)

cusp_model_summary <- summary(cusp_model)

summary(cusp_model, logist=TRUE)
# Call:
#   cusp(formula = follower_sentiment ~ state_sentiment, alpha = alpha ~ 
#          minus_sentiment + comment_variance + frequency + num_nodes + 
#          density + avg_degree_centrality + avg_betweenness_centrality + 
#          avg_closeness_centrality, beta = beta ~ topword_sentiment + 
#          num_edges, data = dada_normalized)
# 
# Deviance Residuals: 
#   Min        1Q    Median        3Q       Max  
# -2.26655   0.01021   0.06522   0.08281   1.69719  
# 
# Coefficients: (1 not defined because of singularities)
# Estimate Std. Error z value Pr(>|z|)    
# a[(Intercept)]                 0.70657    0.07239   9.760  < 2e-16 ***
#   a[minus_sentiment]            -0.26467    0.05026  -5.266 1.39e-07 ***
#   a[comment_variance]            0.06514    0.06350   1.026   0.3050    
# a[frequency]                   0.09200    0.30853   0.298   0.7656    
# a[num_nodes]                  -0.08698    0.30367  -0.286   0.7746    
# a[density]                    -0.06379    0.18737  -0.340   0.7335    
# a[avg_degree_centrality]            NA         NA      NA       NA    
# a[avg_betweenness_centrality]  0.09695    0.08408   1.153   0.2489    
# a[avg_closeness_centrality]    0.01566    0.17352   0.090   0.9281    
# b[(Intercept)]                 4.58850    0.20027  22.911  < 2e-16 ***
#   b[topword_sentiment]           0.13357    0.07979   1.674   0.0941 .  
# b[num_edges]                   0.09979    0.08439   1.183   0.2370    
# w[(Intercept)]                 1.80189    0.03845  46.859  < 2e-16 ***
#   w[state_sentiment]             1.18506    0.02606  45.478  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# 
# Null deviance: 432.543  on 308  degrees of freedom
# Linear deviance: 267.483  on 298  degrees of freedom
# Logist deviance: 137.645  on 297  degrees of freedom
# Delay deviance:  82.834  on 296  degrees of freedom
# 
# R.Squared    logLik npar      AIC     AICc      BIC
# Linear model 0.1315489 -416.1599   11 854.3198 855.2087 895.3866
# Logist model 0.5531015 -313.5132   12 651.0265 652.0805 695.8266
# Cusp model   0.8123406 -127.9259   13 281.8518 283.0857 330.3852
# ---
#   Note: R.Squared for cusp model is Cobb's pseudo-R^2. This value
#       can become negative.
# 
# 	Chi-square test of linear vs. cusp model
# 
# X-squared = 576.5, df = 2, p-value = 0
# 
# Number of optimization iterations: 60

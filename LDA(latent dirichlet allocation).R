#install.packages("tm")
library(tm)
# the data is from Kaggle https://www.kaggle.com/snap/amazon-fine-food-reviews/data  download it by yourself
#
Reviews <- read.csv("C:/Users/sdxin/Desktop/amazon-fine-foods/Reviews.csv", stringsAsFactors=FALSE)
#sales volume is more than 560, four goods
dat.n <-names(which(summary(as.factor(Reviews[,2])) > 560)) #1760 customers 

dataset <- Reviews[which(Reviews[,2]%in%dat.n),]
#head(dataset,100) 
#dim(dataset)

#Corpus
release_corpus <- Corpus(VectorSource(dataset$Text))
#inspect(release_corpus)

#delete numbers and punctuation
library(stringr)
release_corpus <- tm_map(release_corpus, removeNumbers)
release_corpus <- tm_map(release_corpus, removePunctuation)

#remove stop words
release_corpus <- tm_map(release_corpus, removeWords, words=stopwords("en"))



#character to lower
release_corpus <- tm_map(release_corpus, content_transformer(tolower))

#stemming
library(SnowballC)
#release_corpus <- tm_map(release_corpus,stemDocument)

##remove other unnecessary words 
rmwords <- c("they","this","just","even","these","will","dont","much","amazon","amazoncom","give","first",
             "also","think","make","now","want","still","never","lot","got","thought","sure",
             "without","whenever","unlike","somehow","yes","tend","today","the","have","day",
             "not","but","anything")
release_corpus <- tm_map(release_corpus, removeWords, words=rmwords)

tdm <- TermDocumentMatrix(release_corpus)
#tdm

#sparse terms
tdm <- removeSparseTerms(tdm,1-(10/length(release_corpus))) #term-document matrix
#tdm

inspect(tdm[1:20, 1:5])
tdm1=(as.matrix(tdm))
#dim(tdm1)

freq_tdm <- rowSums(tdm1)
sort_tdm <- sort(freq_tdm, decreasing = TRUE) #frequency of words
#sort_tdm 
#summary(sort_tdm)
#inspect(release_corpus)


#LDA model using package
#library(topicmodels)
#dtm <- DocumentTermMatrix(release_corpus)  # document-term matrix
#lda_out <- LDA(dtm,10) 
#posterior_lda <- posterior(lda_out)
#lda_topics <-data.frame(t(posterior_lda$topics))
#lda_topics

#LDA model with my own algorithm
V <- dim(tdm)[1] #number of terms(vocabulary)  743
D <- dim(tdm)[2] #number of documents(reviews) 1760
K <- 5          #number of topics              5

#after the mcmc(collapsed Gibbs sampling) iterations, 3000 iterations are necessary for convergency, bust still not enough
niters <- 2000    #number of iterations of burn in period
aftniters <- 2000 #number of iterations after burn in period, 4000 iterations takes about 4 hours
alpha = 0.001    
alphas = rep(alpha, times = K) 
# Parameter for Dirichlet prior used to find topic distribution over words.
beta = 0.001     
betas = rep(beta, times = V) 

# number of words in each document assigned to topic k = 1:K
# theta, its a fixed size matrix D x K
theta <- matrix(0,nrow=D, ncol=K)
# (phi),number of times word w (unique) is assigned to topic k=1:K
phi <- matrix(0,nrow=K, ncol=V)
# number of words in each document assigned to topic k = 1:K
# the documents containing topic k,its a fixed size matrix D x K
ndkMatrix <- matrix(0,nrow=D, ncol=K)
#number of times word w (unique) is assigned to topic k=1:K
nkwMatrix <- matrix(0,nrow=V, ncol=K)

# number of words assigned to topic k=1:K, one dimensional array.
nk <- rep(0,K)

documents <- list(D)
wordTopics <- list(D) # size of D x size of documents (variable size). The matrix is zagged.

# trace of output to check it is convergent or not
hist_theta = matrix(0,(niters+aftniters),K)   #K=10
hist_phi = matrix(0,(niters+aftniters),V)     #V=743
# record of theta and phi in each iteration
list_theta = list(aftniters)
list_phi = list(aftniters)

library(bayesm)
# randomly initialize topic for each word (theta); [1,K].
for(d in 1:D){
  theta[d,] <- rdirichlet(alphas) 
}
# randomly initialize term for each topic (phi); [1,V].
for(k in 1:K){
  phi[k,] <- rdirichlet(betas) 
}

# d - document index
# w - word index in the document d
for (d in 1:D){
  docWords <- which(tdm1[,d]!=0)  # choose the words occured in document d
  docWordTopics <- list(length(docWords)) # make the word-topic list in document d
  for (wd in 1: length(docWords)) {      # for each word
    # randomly set a topic (1:K)
    wordId <- docWords[wd]        # the ID of the word 
    # draw a sample from a multinomial distribution
    topic <- sample(K, 1)
    docWordTopics[[wd]] <- topic
    ndkMatrix[d,topic] <- ndkMatrix[d,topic] + 1
    nkwMatrix[wordId, topic] <- nkwMatrix[wordId, topic] + 1
    nk[topic] <- nk[topic] + 1
  }
  wordTopics[[d]] <- docWordTopics
}

# iteratations of gibbs sampling, burn in period
for(it in 1:niters) {
  print(c("Iteration: %i, out of %i", it, niters))
  # iterate through all documents and all words in them.
  for (d in 1:D){
    # get list of words and their current topic assignments in document d.
    docWords <- which(tdm1[,d]!=0)
    docWordTopics <- wordTopics[[d]] 
    
    # iterate through all words in the document
    for (wdIdx in 1: length(docWords)) {
      wordId <- docWords[wdIdx]             # the wdIdx word
      wordTopic <- docWordTopics[[wdIdx]]   # the topic assignment of wdIdw
      
      # reduce the count as we are going to assin the topic 
      # and only depend on assignment of topic to all other words.
      ndkMatrix[d, wordTopic] <- ndkMatrix[d, wordTopic] - 1
      nkwMatrix[wordId, wordTopic] <- nkwMatrix[wordId, wordTopic] - 1
      nk[wordTopic] <- nk[wordTopic] - 1
      
      # find the probability of topic k generating the word w, it will be multi nomial
      multkw <- rep(0.0, K)
      for (k in 1:K) {
        multkw[k] <- (ndkMatrix[d,k] + alpha)*(nkwMatrix[wordId,k]+ beta)/(nk[k] + beta*V) # conditional posterior probability of topic assignment
      }
      # sample new topic [1,K] from multinomial distribution and update the topic assignment for objective word.
      wordNewTopic <- sample(K,size=1, prob = multkw)
      docWordTopics[[wdIdx]] <- wordNewTopic
      
      # increment count based on newly assigned topic.
      ndkMatrix[d, wordNewTopic] <- ndkMatrix[d, wordNewTopic] + 1
      nkwMatrix[wordId, wordNewTopic] <- nkwMatrix[wordId, wordNewTopic] + 1
      nk[wordNewTopic] <- nk[wordNewTopic] + 1
    }
    wordTopics[[d]] <- docWordTopics
  } 
  ############## calculate theta and phi matrics ##################################
  # calculate phi
  for(k in 1:K) {
    # calculate normalization factor.
    nzw <- 0
    for (v in 1:V) {
      nzw <- nzw + nkwMatrix[v, k] 
    }
    nzw <- nzw + V*beta
    #calculate phi for each word for the given topic.
    for(v in 1:V) {
      phi[k,v] <- (nkwMatrix[v, k] + beta)/nzw
    }
  }
  # calculate theta
  for(d in 1:D){
    ndz <- 0
    for (k in 1:K){
      ndz <- ndz + ndkMatrix[d, k]
    }
    ndz <- ndz+ K*alpha
    for(k in 1:K) {
      theta[d,k] <- (ndkMatrix[d, k]+alpha)/ndz
    }
  }
  
  #record the trace of theta and phi, it can be used to check the convergency
  hist_theta[it,] <- colMeans(theta)
  hist_phi[it,] <- colMeans(phi)
} #gibbs sampling iteration loop ends here.

# iteratations of gibbs sampling, after burn in period
for(it in 1:aftniters) {
  print(c("Iteration: %i, out of %i", it, niters))
  # iterate through all documents and all words in them.
  for (d in 1:D){
    # get list of words and their current topic assignments in document d.
    docWords <- which(tdm1[,d]!=0)
    docWordTopics <- wordTopics[[d]] 
    
    # iterate through all words in the document
    for (wdIdx in 1: length(docWords)) {
      wordId <- docWords[wdIdx]             # the wdIdx word
      wordTopic <- docWordTopics[[wdIdx]]   # the topic assignment of wdIdw
      
      # reduce the count as we are going to assin the topic 
      # and only depend on assignment of topic to all other words.
      ndkMatrix[d, wordTopic] <- ndkMatrix[d, wordTopic] - 1
      nkwMatrix[wordId, wordTopic] <- nkwMatrix[wordId, wordTopic] - 1
      nk[wordTopic] <- nk[wordTopic] - 1
      
      # find the probability of topic k generating the word w, it will be multi nomial
      multkw <- rep(0.0, K)
      for (k in 1:K) {
        multkw[k] <- (ndkMatrix[d,k] + alpha)*(nkwMatrix[wordId,k]+ beta)/(nk[k] + beta*V) 
        #multkw[k] <- 0.3
      }
      # sample new topic [1,K] from multinomial distribution and update the topic assignment for current word.
      wordNewTopic <- sample(K,size=1, prob = multkw)
      docWordTopics[[wdIdx]] <- wordNewTopic
      
      # increment count based on newly assigned topic.
      ndkMatrix[d, wordNewTopic] <- ndkMatrix[d, wordNewTopic] + 1
      nkwMatrix[wordId, wordNewTopic] <- nkwMatrix[wordId, wordNewTopic] + 1
      nk[wordNewTopic] <- nk[wordNewTopic] + 1
    }
    wordTopics[[d]] <- docWordTopics
  } 
  ############## calculate  theta and phi matrics ##################################
  # calculate phi
  for(k in 1:K) {
    # calculate normalization factor.
    nzw <- 0
    for (v in 1:V) {
      nzw <- nzw + nkwMatrix[v, k] 
    }
    nzw <- nzw + V*beta
    #calculate phi for each word for the given topic.
    for(v in 1:V) {
      phi[k,v] <- (nkwMatrix[v, k] + beta)/nzw
    }
  }
  # calculate theta
  for(d in 1:D){
    ndz <- 0
    for (k in 1:K){
      ndz <- ndz + ndkMatrix[d, k]
    }
    ndz <- ndz+ K*alpha
    for(k in 1:K) {
      theta[d,k] <- (ndkMatrix[d, k]+alpha)/ndz
    }
  }
  #record the trace of theta and phi
  hist_theta[it+niters,] <- colMeans(theta)
  hist_phi[it+niters,] <- colMeans(phi)
  list_theta[[it]] <- theta
  list_phi[[it]] <- phi
} #gibbs sampling iteration loop ends here.
#check the convergency of theta and phi
matplot(hist_theta,type = "l")
matplot(hist_phi[,1:15],type = "l")

#posterior mean
posterior_phi<- apply(simplify2array(list_phi), c(1,2), sum)/aftniters
posterior_theta<- apply(simplify2array(list_theta), c(1,2), sum)/aftniters
#topic_document assignment, to measure which topic is most popular among documents
topic_document_assignment<- apply(posterior_theta,1,which.max)
hist(topic_document_assignment)
#word topic assignment, to show the meaning of each topic
word_topic_assignment<- apply(posterior_phi,2,which.max)
hist(word_topic_assignment)
word_topic_classify<- t(apply(posterior_phi,1,order))
word_topic_classify0<- word_topic_classify[,1:15]

word_topic_classify1<-matrix(0,nrow = 15, ncol = K)
for(k in 1:K){
  word_topic_classify1[,k]<-rownames(tdm)[word_topic_classify0[k,]]
}
colnames(word_topic_classify1)<-colnames(word_topic_classify1, do.NULL = FALSE, prefix = "topic")




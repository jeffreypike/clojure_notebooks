;; # Logistic Regression with Gradient Descent
;; In this notebook we train a logistic regression model from scratch on 
;; synthetic data using gradient descent.

;; ### Setup

(ns logistic_regression
  (:require [nextjournal.clerk :as clerk]
            [uncomplicate.neanderthal
             [native :refer [dv dge]]
             [core :refer [subvector mv xpy entry ax trans sum col mrows dim]]
             [random :refer [rand-normal!]]
             [vect-math :refer [sigmoid]]]
            [tablecloth.api :as tc]))

;; ### Utility Functions
;; First let's define a quality of life function.

(defn ->vec
  "For converting Neanderthal vectors into Clojure vectors"
  [x]
  (into [] x))

;; ### Regression Functions
;; Now let's get in to the real meat of the notebook: the functions that
;; define and train the regression models.  We'll introduce the model itself,
;; a helper function for plotting the decision boundary of a model, and a
;; bare-bones gradient decscent algorithm.

(defn logistic_regression
  "The output of a logistic regression model"
  [params X]
  (sigmoid (xpy (mv X (get params :weights))
                (ax (get params :bias) (dv (repeat (mrows X) 1))))))

(defn decision_boundary
  "The decision boundary line of a logistic regression model"
  [params x]
  (xpy (ax (/ (* -1 (entry (get params :weights) 0))
              (entry (get params :weights) 1))
           x)
       (ax (/ (* -1 (get params :bias))
              (entry (get params :weights) 1))
           (dv (repeat (dim x) 1)))))

(defn step
  "Perform one step of gradient descent"
  [params X y learning_rate]
  {:weights (xpy (get params :weights)
                 (ax (* -1 learning_rate)
                     (mv (trans X)
                         (xpy (logistic_regression params X)
                              (ax -1 y)))))
   :bias (+ (get params :bias)
            (* (* -1 learning_rate)
               (sum (xpy (logistic_regression params X)
                         (ax -1 y)))))})

(defn train
  "Train a logistic regression model with gradient descent"
  ([params X y learning_rate epochs count]
   (if (= epochs count)
     params
     (recur (step params X y learning_rate) X y learning_rate epochs (inc count))))
  ([params X y learning_rate epochs]
   (train params X y learning_rate epochs 0)))

;; ### Generate Random Data
;; First we generate some synthetic data.  For simplicity, we'll generate
;; two Gaussian clouds, one centred at the origin and the other centred 
;; at (1,1).

(def xy (xpy (rand-normal! (dge 100 2)) 
             (dge 100 2 (concat (repeat 50 0) (repeat 50 1) (repeat 50 0) (repeat 50 1)))))
(def labels (dv (concat (repeat 50 0) (repeat 50 1))))

;; For this notebook, creating an R-like (or pandas-like) dataframe
;; is completely unnecessary, but let's do it anyways!

(clerk/table (tc/dataset {:x (->vec (col xy 0)) 
                          :y (->vec (col xy 1)) 
                          :label (->vec labels)}))

;; To cap this section off, let's plot the data we created.

(clerk/plotly {:data [{:x (->vec (subvector (col xy 0) 0 50))
                       :y (->vec (subvector (col xy 1) 0 50))
                       :name "Class 0"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (subvector (col xy 0) 50 50))
                       :y (->vec (subvector (col xy 1) 50 50))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"}]})

;; ### Initialize Random Parameters
;; Since we're using gradient descent, let's start our model off by
;; initializing some random parameters... this should produce a terrible
;; model, but hey, you never know!

(def params_init
  {:weights (rand-normal! (dv 2))
   :bias (entry (rand-normal! (dv 1)) 0)})

;; Let's take a look at just how bad this model is:
(clerk/plotly {:data [{:x (->vec (subvector (col xy 0) 0 50))
                       :y (->vec (subvector (col xy 1) 0 50))
                       :name "Class 0"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (subvector (col xy 0) 50 50))
                       :y (->vec (subvector (col xy 1) 50 50))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (col xy 0))
                       :y (->vec (decision_boundary params_init (col xy 0)))
                       :name "Decision Boundary"}]})

;; ### Train the Model
;; Now let's train the model! We'll use a learning rate of 0.0001 and train
;; for 1000 epochs.  I'm sure more efficient choices could have been made,
;; but this will get the job done.

(def params_optimal (train params_init xy labels 0.0001 1000))

;; Finally, let's admire our work!

(clerk/plotly {:data [{:x (->vec (subvector (col xy 0) 0 50))
                       :y (->vec (subvector (col xy 1) 0 50))
                       :name "Class 0"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (subvector (col xy 0) 50 50))
                       :y (->vec (subvector (col xy 1) 50 50))
                       :name "Class 1"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec (col xy 0))
                       :y (->vec (decision_boundary params_optimal (col xy 0)))
                       :name "Decision Boundary"}]})
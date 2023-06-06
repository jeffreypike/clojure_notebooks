;; # Linear Regression with Gradient Descent
;; Welcome to the first notebook in a mini-series on data science in Clojure!  These notebooks grew out of my efforts to familiarize
;; myself with the data science tools that exist in the Clojure community.  My hope is that they can provide a helpful
;; starting point for anyone that is new to Clojure, new to data science altogether, or both!

;; ## Why Clojure?
;; Clojure is a beautiful language coming from the lisp family, where the overarching philosophy is 'code is data', a philosophy that
;; goes a long way in data science!  The main obstacle to the use of lisps for data science lies in the lack of a developed ecosystem.
;; It is simply too much to ask to reinvent the wheel in the data science world, as there are a plethora of complex models and algorithms
;; that could be valuable tools and let's be honest... we want to try them all!  Luckily, Clojure's relationship with Java provides a
;; built-in solution to this problem by giving us access to the Java data science ecosystem practically for free.  In short, Clojure
;; brings the best of both worlds by providing the data-first design philosophy of lisps along with the well-developed package base of
;; Java, making it the perfect language for data science!

;; ## Setup
;; There are a few packages that we'll make use of to make our regression dreams come true:
;; * [Clerk](https://github.com/nextjournal/clerk): The tool used to generate this very notebook!  If you're used to working
;; in Jupyter notebooks, then you're going to love Clerk because it provides that same experience without asking you to
;; leave your favourite editor.  Seriously, give it a try... it's awesome!
;; * [Neanderthal](https://neanderthal.uncomplicate.org/): The engine that makes it all possible.  Neanderthal is designed to
;; give us highly optimized linear algebra in Clojure.  For those of us coming from Python, this is our NumPy equivalent 
;; (but even faster!).  Since we're doing everything from scratch, this is the only package we truly _need_.
;; * [SciCloj](https://github.com/scicloj/scicloj.ml): After implementing our regression model directly, we'll also show how we can solve
;; the same problem using an efficient modeling package.  For the Python users, this is our SciKit Learn, though it also comes with
;; a wrapper around [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset), which is our Pandas equivalent.

(ns linear_regression
  (:require [nextjournal.clerk :as clerk] 
            [uncomplicate.neanderthal
             [native :refer [dv dge]]
             [core :refer [axpby! axpy mv xpy entry ax dot dim trans sum col mrows]]
             [random :refer [rand-normal!]]
             [real :refer [entry!]]]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [scicloj.ml.dataset :as ds]))

;; ### Utility Functions
;; We'll begin by defining a few quality of life functions.  The main datatype we're going to use throughout this notebook
;; is Neanderthal's `RealBlockVector`, which is optimized for computations.  However, many of the other packages we want
;; to make use of (tablecloth, plotly, etc) expect a regular old Clojure vector, so let's kick things off by defining a 
;; helper function to transform a Neanderthal vector to something that's readable by packages that don't know Neanderthal exists.

(defn ->vec 
  "For converting Neanderthal vectors into Clojure vectors"
  [x] 
  (into [] x))

;; Next up, it looks like Neanderthal doesn't come with a built-in way to take the mean of a vector. It does however have a function
;; that returns the sum of the entries (`sum`), as well as one that returns the dimension (`dim`), so defining our own mean function is straightforward
;; and should help clean up our code a bit.

(defn mean
  "Take the mean of a Neanderthal vector"
  [x]
  (* (/ 1 (dim x)) (sum x)))

;; The last utility function we'll define simply returns a vector or matrix full of ones.  This will be helpful for
;; [vector broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), which Neanderthal doesn't handle
;; for us.  We'll do this by changing all the entries of a newly initialized object (whcih is full of zeros by default)
;; to one.  Note that in Neanderthal, an exlamation point at the end of a function name indicates that the function is
;; overwriting one of its inputs (use with caution!).  For example, `entry` _gets_ the entries of a vector or matrix, while
;; `entry!` _sets_ them.

(defn ones
  "Create a vector or matrix of ones"
  ([n]
  (entry! (dv n) 1))
  ([n m]
   (entry! (dge n m) 1)))

;; ### Regression Functions
;; Now let's get in to the real meat of the notebook: the functions that define and train the regression models.
;; _Linear regression_ is really a fancy word for the "line of best fit" (okay, okay... _hyperplane_ of best fit),
;; in other words it is simply the linear equation that comes closest to representing the trend found in our data.
;; Mathematically the model itself is just a linear equation.  That is, for a single sample $\vec x$ from our data, by
;; a linear regression applied to $\vec x$ we just mean the quantity 

;; $$ \ell (\vec x) = \vec x \cdot \vec w + b $$.

;; Here $\vec w$ is a vector of _weights_ (so called because they determine how much to weigh each quantity being
;; measured in our data when computing $\ell$), and $b$ is a scalar called the _bias_ (or sometimes the _intercept_).
;; In general, we are not interested in applying our model just to one sample of our data, but rather to a large
;; number of samples all at once.  We can compute this efficiently by stacking our individual samples as the
;; rows of a matrix $X$, and taking advantage of matrix multiplication.  If we have $n$ data points, each comprised
;; of $p$ measurements, then this takes the form:

;; $$ \ell (X) = \begin{pmatrix} x_{11} & \cdots & x_{1p} \\ \vdots & \ddots & \vdots \\ x_{n1} & \cdots & x_{np} \end{pmatrix} \begin{pmatrix} w_1 \\ \vdots \\ w_p \end{pmatrix} + \begin{pmatrix} b \\ \vdots \\ b \end{pmatrix} $$

;; Together the weights $w_1,\dots, w_p$ and the bias $b$ form the _parameters_ of the model.  Of course, the equation
;; above assumes that we _know_ the parameters.  In fact, the problem statement of linear regression is to "determine"
;; the best parameters for $\ell$.  So far we have been thinking of $\ell$ as a function of $X$, but really we have 
;; two distinct phases that change the way we think about $\ell$.  First, before we've determined the optimal parameters,
;; then we think of the data $X$ as given and $\ell$ is a function of the parameters; $\ell = \ell(\vec w, b)$ (this is
;; known as the _training_ or _fitting_ phase)
(defn linear_regression
  "A typical linear equation"
  [params X]
  (axpy (get params :bias) (ones (mrows X))
        (mv X (get params :weights))))

(defn r2
  "The R-squared of a linear regression"
  [Yhat Y]
  (- 1 (/ (dot (axpy -1 Y Yhat)
               (axpy -1 Y Yhat))
          (dot (axpy (* -1 (mean Y)) (ones (dim Y)) Y)
               (axpy (* -1 (mean Y)) (ones (dim Y)) Y)))))

(defn step
  "Perform one step of gradient descent"
  [params X y learning_rate]
  {:weights (axpy (* -1 learning_rate)
                  (mv (trans X)
                      (axpy -1 y (linear_regression params X)))
                  (get params :weights))
   :bias (+ (get params :bias)
            (* (* -1 learning_rate)
               (sum (axpy -1 y (linear_regression params X)))))})

(defn train
  "Train a linear regression model with gradient descent"
  ([params X y learning_rate epochs count]
   (if (= epochs count)
     params 
     (recur (step params X y learning_rate) X y learning_rate epochs (inc count))))
   ([params X y learning_rate epochs]
    (train params X y learning_rate epochs 0)))

;; ## Univariate Regression
;; We'll start with the single variable case so that we can easily
;; visualize what's going on, but the goal is that our code generalizes
;; seamlessly to the multivariate setting.

;; ### Generate Random Data
;; First we generate some synthetic data.  We'll use the line y = 2x as 
;; our baseline and add some random noise.

(def x (rand-normal! (dv 100)))
(def y (axpby! 2 x 1 (rand-normal! (dv 100))))

;; And to cap this section off, let's plot the data we created.s

(clerk/plotly {:data [{:x (->vec x) 
                       :y (->vec y) 
                       :type "scatter" 
                       :mode "markers"}]})

;; ### Initialize Random Parameters
;; Since we're using gradient descent, let's start our model off by
;; initializing some random parameters.  In the 1D case, both the
;; weight and the bias are scalar, but since we want to easily generalize
;; to multivariate regression, let's make the weight a 1D vector.

(def params_init 
  {:weights (rand-normal! (dv 1)) 
   :bias (entry (rand-normal! (dv 1)) 0)})

;; Just for fun, let's create a model with these random weights.
(def yhat_rdm (linear_regression params_init (dge 100 1 x)))

;; How does this model do?  Unless we're very lucky, this R-squared should
;; be quite far from 1.
(r2 yhat_rdm y)

;; Let's see just how bad this model is:
(clerk/plotly {:data [{:x (->vec x)
                       :y (->vec y)
                       :name "data"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec x)
                       :y (->vec yhat_rdm)
                       :name "regression"}]})

;; ### Train the Model
;; Now let's train the model!  We'll use a learning rate of 0.0001 and
;; train for 1000 epochs.  Both of these numbers were arbitrarily chosen,
;; I'm sure we could have made significantly more efficient choices!  We 
;; will hopefully end up with a weight close to 2 and a bias close to 0,
;; but it won't be exact because of the noise.

(def params_optimal (train params_init (dge 100 1 x) y 0.0001 1000))

;; Awesome, let's create a model with these parameters!

(def yhat (linear_regression params_optimal (dge 100 1 x)))

;; The R-squared of this model will be much closer to 1 than that of the random model.

(r2 yhat y)

;; Finally, let's admire our work!

(clerk/plotly {:data [{:x (->vec x)
                       :y (->vec y)
                       :name "data"
                       :type "scatter"
                       :mode "markers"}
                      {:x (->vec x)
                       :y (->vec yhat)
                       :name "regression"}]})

;; ## Multivariate Regression
;; We've seen that our code works for the single-variable case, now let's
;; see if it can also handle multiple variables!

;; ### Generate Random Data
;; Once again we'll just use synthetic data.  We'll add some noise to
;; a set of sample points from the plane z = 3x + 5y.
(def xy (rand-normal! (dge 100 2)))
(def xx (col xy 0))
(def yy (col xy 1))
(def zz (xpy (xpy (ax 3 xx) (ax 5 yy)) (rand-normal! (dv 100))))

;; Just for fun, let's throw this in to a table!

(clerk/table (ds/dataset {:x (->vec xx) :y (->vec yy) :z (->vec zz)}))

;; And let's take a look at our data.

(clerk/plotly {:data [{:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zz)
                       :type "scatter3d"
                       :mode "markers"
                       :marker {:size 2}}]})

;; ### Initialize Random Parameters
;; Just like in the univariate case, we'll start with a random model and
;; use gradient descent to train it.  So let's initialize the parameters
;; randomly.

(def params_init_3d
  {:weights (rand-normal! (dv 2))
   :bias (entry (rand-normal! (dv 1)) 0)})

;; Computing the model...

(def zzhat_rdm (linear_regression params_init_3d xy))

;;... should yield a poor R-squared.
(r2 zzhat_rdm zz)

;; Let's take a look at how poorly this model fits the data.

(clerk/plotly {:data [{:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zz)
                       :type "scatter3d"
                       :mode "markers"
                       :marker {:size 2}}
                      {:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zzhat_rdm)
                       :type "mesh3d"}]})

;; ## Train the Model

;; Now let's whip those parameters into shape!  We'll go with the same
;; hyperparameters as we did earlier.  Hopefully this yields a bias close
;; to 0 and weights close to [3, 5].

(def params_optimal_3d (train params_init_3d xy zz 0.0001 1000))

;; Building our optimal model...

(def zzhat (linear_regression params_optimal_3d xy))

;; ... yields a great R-squared!

(r2 zzhat zz)

;; And to cap this notebook off, let's take a look at our beautiful 3D model!

(clerk/plotly {:data [{:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zz)
                       :type "scatter3d"
                       :mode "markers"
                       :marker {:size 2}}
                      {:x (->vec xx)
                       :y (->vec yy)
                       :z (->vec zzhat)
                       :type "mesh3d"}]})

;; ## Using SciCloj

;; We can also use SciCloj.  SciCloj prefers dataset structures as inputs.  Preview:

(clerk/table (ds/dataset {:x (->vec x) :y (->vec y)}))

(def pipe-fn
  (ml/pipeline
   (mm/set-inference-target :y)
   {:metamorph/id :model}
   (mm/model {:model-type :smile.regression/ordinary-least-square})))

(def trained-model
  (pipe-fn {:metamorph/data (ds/dataset {:x (->vec x) :y (->vec y)})
            :metamorph/mode :fit}))

(ml/explain (:model trained-model))
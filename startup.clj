(ns startup
  (:require [nextjournal.clerk :as clerk]))

;; start Clerk's built-in webserver on the default port 7777, opening the browser when done
(clerk/serve! {:browse? true})

;; let Clerk watch the given `:paths` for changes
(clerk/serve! {:watch-paths ["notebooks" "src"]})

;; build a static html of a given notebook
(clerk/build! {:paths ["notebooks/logistic_regression.clj"]})
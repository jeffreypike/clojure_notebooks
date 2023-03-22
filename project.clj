(defproject testing "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [io.github.nextjournal/clerk "0.13.838"]
                 [scicloj/tablecloth "7.000-beta-27"]
                 [uncomplicate/neanderthal "0.46.0"]
                 [techascent/tech.ml.dataset "7.000-beta-30"]
                 [aerial.hanami "0.17.0"]]
  :repl-options {:init-ns testing.core})

(defproject notebooks "0.1.0-SNAPSHOT"
  :description "Machine learning notebooks in Clojure"
  :url "https://jeffreypike.github.io/clojure_notebooks"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [io.github.nextjournal/clerk "0.13.838"]
                 [scicloj/tablecloth "7.000-beta-27"]
                 [uncomplicate/neanderthal "0.46.0"]
                 [org.bytedeco/mkl-platform-redist "2022.2-1.5.8"]
                 [techascent/tech.ml.dataset "7.000-beta-30"]]
  :repl-options {:init-ns notebooks.core}
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"
                       "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"
                       "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"])
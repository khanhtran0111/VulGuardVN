// VulGuard v2 - batch CPG fact extraction (plan P2 steps 1, 3, 4 and the facts needed by V1-V3 in §6.5).
// Usage: joern --script extract.sc --param inputDir=<dir of N.c files> --param outFile=<jsonl> \
//              --param sinkApis=a,b,c --param allocApis=... --param sourceApis=... --param maxSinks=30 --param maxSlice=400
//
// One JSON line per input file:
//   {id, ok, method, n_nodes, cyclomatic, params, sinks[{line,kind,name,code,vars}], slices[[line..]],
//    lines{L:{calls,deref,index,ids,tainted,dom}}, conds[{line,code,ids,fields}]}
// V1 = sink line has the call / deref / index on the variable      (lines[L].calls / deref / index)
// V2 = variable reachable from a parameter or source via DDG       (lines[L].tainted)
// V3 = conditions that dominate or control the sink line           (lines[L].dom -> conds)

import io.shiftleft.semanticcpg.language.*
import io.joern.dataflowengineoss.language.*
import io.shiftleft.codepropertygraph.generated.nodes.*
import scala.collection.mutable

def lineOf(n: StoredNode): Int = n match {
  case a: AstNode => a.lineNumber.map(_.toInt).getOrElse(-1)
  case _ => -1
}

def clip(s: String, n: Int): String = if (s.length <= n) s else s.substring(0, n) + "..."

def identNames(n: AstNode): List[String] =
  (n :: n.ast.l).collect { case i: Identifier => i.name }.distinct

def strArr(xs: Iterable[String]): ujson.Arr = ujson.Arr.from(xs.map(ujson.Str(_)))
def intArr(xs: Iterable[Int]): ujson.Arr = ujson.Arr.from(xs.map(x => ujson.Num(x.toDouble)))

@main def exec(inputDir: String, outFile: String, sinkApis: String, allocApis: String, sourceApis: String,
               maxSinks: Int = 30, maxSlice: Int = 400) = {
  val sinkSet = sinkApis.split(",").map(_.trim).filter(_.nonEmpty).toSet
  val allocSet = allocApis.split(",").map(_.trim).filter(_.nonEmpty).toSet
  val sourceSet = sourceApis.split(",").map(_.trim).filter(_.nonEmpty).toSet
  val derefOps = Set("<operator>.indirection", "<operator>.indirectFieldAccess", "<operator>.pointerShift")
  val arithOps = Set("<operator>.multiplication", "<operator>.addition", "<operator>.shiftLeft", "<operator>.subtraction")
  val writer = new java.io.PrintWriter(new java.io.FileWriter(outFile))

  importCode.c(inputDir)

  val files = new java.io.File(inputDir).listFiles().filter(f => f.getName.endsWith(".c") || f.getName.endsWith(".cpp"))
  val methodsByFile: Map[String, List[Method]] = cpg.method.isExternal(false).l
    .filterNot(m => m.name.startsWith("<"))
    .groupBy(m => new java.io.File(m.filename).getName)

  for (file <- files.sortBy(_.getName)) {
    val stem = file.getName.replaceAll("\\.(c|cpp)$", "")
    val candidates = methodsByFile.getOrElse(file.getName, Nil)
    if (candidates.isEmpty) {
      writer.println(ujson.write(ujson.Obj("id" -> stem, "ok" -> false, "error" -> "no_method")))
    } else {
      try {
        val m = candidates.maxBy(_.ast.size)
        val calls = m.call.l
        val lines = mutable.Map[Int, mutable.Map[String, mutable.LinkedHashSet[String]]]()
        def lf(l: Int, key: String): mutable.LinkedHashSet[String] =
          lines.getOrElseUpdate(l, mutable.Map()).getOrElseUpdate(key, mutable.LinkedHashSet())

        // ---- per-line facts (V1)
        m.ast.isIdentifier.foreach(i => if (lineOf(i) > 0) lf(lineOf(i), "ids") += i.name)
        calls.foreach { c =>
          val l = lineOf(c)
          if (l > 0) {
            if (!c.name.startsWith("<operator>")) lf(l, "calls") += c.name
            if (c.name == "<operator>.indirectIndexAccess") {
              c.argument.l.foreach(a => identNames(a).foreach(n => lf(l, "index") += n))
              c.argument.l.headOption.foreach(a => identNames(a).foreach(n => lf(l, "deref") += n))
            }
            if (derefOps.contains(c.name)) c.argument.l.headOption.foreach(a => identNames(a).foreach(n => lf(l, "deref") += n))
          }
        }

        // ---- candidate sinks (P2 step 3)
        case class Sink(line: Int, kind: String, name: String, code: String, vars: List[String], node: Call, prio: Int)
        val sinks = mutable.ArrayBuffer[Sink]()
        calls.foreach { c =>
          val l = lineOf(c)
          if (l > 0) {
            if (sinkSet.contains(c.name)) {
              sinks += Sink(l, "api", c.name, clip(c.code, 200), identNames(c).filterNot(_ == c.name), c, 0)
              if (allocSet.contains(c.name) && c.argument.ast.isCall.exists(x => arithOps.contains(x.name)))
                sinks += Sink(l, "size_arith", c.name, clip(c.code, 200), identNames(c), c, 1)
            } else if (c.name == "<operator>.indirectIndexAccess") {
              sinks += Sink(l, "index", "[]", clip(c.code, 200), identNames(c), c, 2)
            } else if (c.name == "<operator>.indirection" || c.name == "<operator>.indirectFieldAccess") {
              sinks += Sink(l, "deref", if (c.name.endsWith("indirection")) "*" else "->", clip(c.code, 200), identNames(c), c, 3)
            }
          }
        }
        val chosen = sinks.groupBy(s => (s.line, s.kind, s.name)).values.map(_.head).toList
          .sortBy(s => (s.prio, s.line)).take(maxSinks).sortBy(_.line)

        // ---- taint: forward DDG reachability from parameters and source calls (V2)
        val tainted = mutable.Set[Long]()
        val queue = mutable.Queue[StoredNode]()
        m.parameter.foreach(p => queue.enqueue(p))
        calls.filter(c => sourceSet.contains(c.name)).foreach { c => queue.enqueue(c); c.argument.foreach(a => queue.enqueue(a)) }
        var steps = 0
        while (queue.nonEmpty && steps < 20000) {
          val n = queue.dequeue()
          steps += 1
          if (!tainted.contains(n.id)) {
            tainted += n.id
            n._reachingDefOut.foreach(x => queue.enqueue(x))
            n match {
              case e: Expression => e.astParent match {
                case p: Call => queue.enqueue(p)
                case _ =>
              }
              case _ =>
            }
          }
        }
        m.ast.isIdentifier.foreach(i => if (tainted.contains(i.id) && lineOf(i) > 0) lf(lineOf(i), "tainted") += i.name)
        calls.filter(c => tainted.contains(c.id)).foreach { c =>
          c.argument.isIdentifier.foreach(i => if (lineOf(i) > 0) lf(lineOf(i), "tainted") += i.name)
        }

        // ---- guarding conditions (V3): conditions that dominate or control a line
        val conds = m.controlStructure.condition.l
        val condIndexOfNode = mutable.Map[Long, Int]()
        conds.zipWithIndex.foreach { case (cond, idx) =>
          condIndexOfNode(cond.id) = idx
          cond.ast.foreach(x => condIndexOfNode.getOrElseUpdate(x.id, idx))
        }
        val condJson = conds.map { cond =>
          ujson.Obj("line" -> lineOf(cond), "code" -> clip(cond.code, 200), "ids" -> strArr(identNames(cond)),
            "fields" -> strArr(cond.ast.isCall.filter(x => x.name.endsWith("FieldAccess")).code.l.distinct))
        }
        val repByLine = calls.filter(c => lineOf(c) > 0).groupBy(lineOf).view.mapValues(cs => cs.maxBy(_.ast.size)).toMap
        val domByLine = mutable.Map[Int, List[Int]]()
        repByLine.foreach { case (l, rep) =>
          val own = condIndexOfNode.get(rep.id)
          val viaDom = rep.dominatedBy.l.flatMap(x => condIndexOfNode.get(x.id))
          val viaCdg = rep.controlledBy.l.flatMap(x => condIndexOfNode.get(x.id))
          domByLine(l) = (viaDom ++ viaCdg).distinct.filterNot(i => own.contains(i)).sorted
        }

        // ---- backward slices per sink (P2 step 4): DDG + control dependence
        val sliceJson = chosen.map { s =>
          val seen = mutable.Set[Long]()
          val q = mutable.Queue[StoredNode](s.node)
          val out = mutable.SortedSet[Int]()
          while (q.nonEmpty && seen.size < maxSlice) {
            val n = q.dequeue()
            if (!seen.contains(n.id)) {
              seen += n.id
              if (lineOf(n) > 0) out += lineOf(n)
              n._reachingDefIn.foreach(x => q.enqueue(x))
              n match {
                case c: CfgNode => c.controlledBy.foreach(x => q.enqueue(x))
                case _ =>
              }
              n match {
                case c: Call => c.argument.foreach(x => q.enqueue(x))
                case _ =>
              }
            }
          }
          m.lineNumber.foreach(l => out += l.toInt)
          intArr(out)
        }

        val linesJson = ujson.Obj()
        val allLines = (lines.keySet ++ domByLine.keySet).toList.sorted
        allLines.foreach { l =>
          val facts = lines.getOrElse(l, mutable.Map())
          val o = ujson.Obj()
          Seq("calls", "deref", "index", "ids", "tainted").foreach(k => o(k) = strArr(facts.getOrElse(k, Nil)))
          o("dom") = intArr(domByLine.getOrElse(l, Nil))
          linesJson(l.toString) = o
        }
        val obj = ujson.Obj(
          "id" -> stem, "ok" -> true, "method" -> m.name, "n_nodes" -> m.ast.size,
          "cyclomatic" -> (m.controlStructure.size + 1), "params" -> strArr(m.parameter.name.l),
          "line_start" -> m.lineNumber.map(_.toInt).getOrElse(-1), "line_end" -> m.lineNumberEnd.map(_.toInt).getOrElse(-1),
          "sinks" -> ujson.Arr.from(chosen.map(s => ujson.Obj("line" -> s.line, "kind" -> s.kind, "name" -> s.name,
            "code" -> s.code, "vars" -> strArr(s.vars)))),
          "slices" -> ujson.Arr.from(sliceJson), "lines" -> linesJson, "conds" -> ujson.Arr.from(condJson))
        writer.println(ujson.write(obj))
      } catch {
        case e: Throwable =>
          writer.println(ujson.write(ujson.Obj("id" -> stem, "ok" -> false, "error" -> clip(e.toString, 300))))
      }
    }
    writer.flush()
  }
  writer.close()
}

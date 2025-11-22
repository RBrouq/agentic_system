# LangGraph integration adapted to your Essay Agent code
try:
    from langgraph import Graph, Node
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False

def build_graph(analyzer, planner, writer, critic, searcher, judge=None):
    """
    Builds a LangGraph for the essay agent workflow:
    - analyzer -> planner -> searcher -> writer -> critic
    - loops critic -> searcher / writer for revisions
    - optional judge LLM at the end
    """
    if not HAS_LANGGRAPH:
        raise RuntimeError('langgraph not installed')
    
    g = Graph('essay-agent')
    
    # Define nodes with the agents you provided
    analyzer_node = Node('analyzer', lambda topic: analyzer.run(topic))
    planner_node = Node('planner', lambda analysis, sources=None: planner.run(analysis, sources))
    searcher_node = Node('searcher', lambda query, num=3: searcher.search(query, num=num))
    writer_node = Node('writer', lambda plan, sources=None, hints="": writer.run(plan, sources, hints))
    critic_node = Node('critic', lambda draft: critic.run(draft))
    
    # Optional Judge node for final quality assessment
    judge_node = Node('judge', lambda draft: judge.run(draft)) if judge else None
    
    # Add nodes to graph
    g.add_nodes([analyzer_node, planner_node, searcher_node, writer_node, critic_node])
    if judge_node:
        g.add_nodes([judge_node])
    
    # Connect nodes: linear workflow + revision loops
    g.connect(analyzer_node, planner_node)      # analyzer → planner
    g.connect(planner_node, searcher_node)     # planner → searcher
    g.connect(searcher_node, writer_node)      # searcher → writer
    g.connect(writer_node, critic_node)        # writer → critic
    
    # Loops for revision if critic suggests improvements
    g.connect(critic_node, searcher_node)      # critic → searcher (new info)
    g.connect(critic_node, writer_node)        # critic → writer (revise draft)
    
    # Final judge (optional)
    if judge_node:
        g.connect(writer_node, judge_node)
    
    return g


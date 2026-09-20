"""Clean-room batch boolean rule-tree evaluator for capa-style rule sets.

Written fresh from the documented semantics of a boolean rule language
(and / or / not / N-or-more / count-range over a feature map). No
third-party Mojo code is used or adapted.

The rule set is compiled once into flat, postfix-ordered node arrays
(children always precede their parent). One `capamojo_eval` call then
evaluates EVERY rule against a block of up to 64 scopes at once: the
value of each node across the 64 scopes is one 64-bit word, so the
boolean combination of subtrees is a single bitwise instruction per
node per block. Feature leaves read per-(leaf, scope) occurrence counts
prepared by the caller; "match" leaves reference the already-computed
result word of an earlier rule (rules arrive in topological order), or
the OR of a group of rule words (a namespace match).

Exported C ABI (v1):

    int32_t  capamojo_abi_version(void)
    void*    capamojo_ruleset_create(n_rules, rule_node_offsets,
                                     node_op, node_child_offsets,
                                     node_children, node_arg,
                                     node_range_min, node_range_max,
                                     n_count_leaves, n_match_groups,
                                     match_group_offsets, match_group_rules)
    int32_t  capamojo_eval(handle, n_scopes, leaf_counts,
                           leaf_present_words, out_rule_words)
    void     capamojo_ruleset_destroy(handle)

Node op codes:
    1 AND      word = & of children words (all-lanes word if no children)
    2 OR       word = | of children words (zero word if no children)
    3 NOT      word = ~child word, masked to the active lanes
    4 SOME     word = per-lane popcount(children) >= arg (threshold); a zero
               threshold is true on all lanes when there ARE children and
               false when there are none (reference short-circuit semantics)
    5 RANGE    word = per-lane min <= leaf_counts[arg][lane] <= max
    6 LEAF     arg < n_count_leaves: word = leaf_present_words[arg]
               arg >= n_count_leaves: word = leaf_present_words[arg] | the OR
                       of the match group's rule words

`leaf_counts` is leaf-major: leaf_counts[leaf * n_scopes + lane], one
non-negative int32 per (leaf, scope) holding the feature's location count
(0 both when absent and when present with an empty location set).
`leaf_present_words` holds one 64-bit word per slot (count leaves plus one
slot per match group): bit `lane` is set when the feature key is present in
that scope's map at all — a distinction that matters because the reference
semantics treat a feature with an empty location set as present for leaf
matching but as count 0 for ranges. For pre-resolved "special" leaves
(substring/regex/bytes) the caller sets the presence bit when the scan
succeeded; for match-group slots it sets the bit when a ("match", name)
feature is present in the input map. `out_rule_words` receives one word per
rule: bit `lane` set means the rule matched that scope. Lanes >= n_scopes
are always reported 0.
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

comptime OP_AND: Int32 = 1
comptime OP_OR: Int32 = 2
comptime OP_NOT: Int32 = 3
comptime OP_SOME: Int32 = 4
comptime OP_RANGE: Int32 = 5
comptime OP_LEAF: Int32 = 6

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; the library owns what it allocates).
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]


struct RuleSet(Copyable, Movable):
    """Owned, native copy of one compiled rule set (flat node arrays)."""

    var n_rules: Int64
    var total_nodes: Int64
    var total_children: Int64
    var n_count_leaves: Int64
    var n_match_groups: Int64
    var total_match_refs: Int64
    var max_rule_nodes: Int64
    var rule_node_offsets: I64Ptr  # [n_rules + 1]
    var node_op: I32Ptr  # [total_nodes]
    var node_child_offsets: I64Ptr  # [total_nodes + 1]
    var node_children: I32Ptr  # [total_children] global node indices
    var node_arg: I32Ptr  # [total_nodes] leaf id / match-group base / threshold
    var node_range_min: U64Ptr  # [total_nodes] RANGE only
    var node_range_max: U64Ptr  # [total_nodes] RANGE only
    var match_group_offsets: I64Ptr  # [n_match_groups + 1]
    var match_group_rules: I32Ptr  # [total_match_refs]
    var scratch: U64Ptr  # [max_rule_nodes] per-rule node words, reused per eval

    def __init__(
        out self,
        n_rules: Int64,
        total_nodes: Int64,
        total_children: Int64,
        n_count_leaves: Int64,
        n_match_groups: Int64,
        total_match_refs: Int64,
        max_rule_nodes: Int64,
        rule_node_offsets: I64Ptr,
        node_op: I32Ptr,
        node_child_offsets: I64Ptr,
        node_children: I32Ptr,
        node_arg: I32Ptr,
        node_range_min: U64Ptr,
        node_range_max: U64Ptr,
        match_group_offsets: I64Ptr,
        match_group_rules: I32Ptr,
        scratch: U64Ptr,
    ):
        self.n_rules = n_rules
        self.total_nodes = total_nodes
        self.total_children = total_children
        self.n_count_leaves = n_count_leaves
        self.n_match_groups = n_match_groups
        self.total_match_refs = total_match_refs
        self.max_rule_nodes = max_rule_nodes
        self.rule_node_offsets = rule_node_offsets
        self.node_op = node_op
        self.node_child_offsets = node_child_offsets
        self.node_children = node_children
        self.node_arg = node_arg
        self.node_range_min = node_range_min
        self.node_range_max = node_range_max
        self.match_group_offsets = match_group_offsets
        self.match_group_rules = match_group_rules
        self.scratch = scratch


def _copy_i64(src: I64Ptr, count: Int64) -> I64Ptr:
    var dst = unsafe_alloc[Int64](Int(count))
    if count > 0:
        unsafe_memcpy(dest=dst, src=src, count=Int(count))
    return dst


def _copy_i32(src: I32Ptr, count: Int64) -> I32Ptr:
    var dst = unsafe_alloc[Int32](Int(count))
    if count > 0:
        unsafe_memcpy(dest=dst, src=src, count=Int(count))
    return dst


def _copy_u64(src: U64Ptr, count: Int64) -> U64Ptr:
    var dst = unsafe_alloc[UInt64](Int(count))
    if count > 0:
        unsafe_memcpy(dest=dst, src=src, count=Int(count))
    return dst


@export
def capamojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def capamojo_ruleset_create(
    n_rules: Int64,
    rule_node_offsets: I64Ptr,
    node_op: I32Ptr,
    node_child_offsets: I64Ptr,
    node_children: I32Ptr,
    node_arg: I32Ptr,
    node_range_min: U64Ptr,
    node_range_max: U64Ptr,
    n_count_leaves: Int64,
    n_match_groups: Int64,
    match_group_offsets: I64Ptr,
    match_group_rules: I32Ptr,
) abi("C") -> Handle:
    """Copy the caller's flat arrays into a native rule set; NULL on invalid input."""
    if n_rules <= 0 or n_count_leaves < 0 or n_match_groups < 0:
        return None

    var total_nodes = rule_node_offsets[unsafe_offset=Int(n_rules)]
    if total_nodes < n_rules:  # every rule has at least a root node
        return None
    var total_children = node_child_offsets[unsafe_offset=Int(total_nodes)]
    if total_children < 0:
        return None
    var total_match_refs = Int64(0)
    if n_match_groups > 0:
        total_match_refs = match_group_offsets[unsafe_offset=Int(n_match_groups)]
        if total_match_refs < 0:
            return None

    # Validate structure: postfix order (children precede parents), in-range
    # leaf/group/rule references, sane ops. Reject anything else.
    var max_rule_nodes = Int64(0)
    for r in range(Int(n_rules)):
        var start = rule_node_offsets[unsafe_offset=r]
        var end = rule_node_offsets[unsafe_offset=r + 1]
        if end <= start:
            return None
        if end - start > max_rule_nodes:
            max_rule_nodes = end - start
    for i in range(Int(total_nodes)):
        var op = node_op[unsafe_offset=i]
        if op < OP_AND or op > OP_LEAF:
            return None
        var c0 = node_child_offsets[unsafe_offset=i]
        var c1 = node_child_offsets[unsafe_offset=i + 1]
        if c1 < c0:
            return None
        if op == OP_NOT and c1 - c0 != 1:
            return None
        if (op == OP_LEAF or op == OP_RANGE) and c1 - c0 != 0:
            return None
        for c in range(Int(c0), Int(c1)):
            var child = Int(node_children[unsafe_offset=c])
            if child < 0 or child >= i:  # postfix: children strictly earlier
                return None
        var arg = Int(node_arg[unsafe_offset=i])
        if op == OP_LEAF:
            if arg < 0:
                return None
            if arg >= Int(n_count_leaves):
                var group = arg - Int(n_count_leaves)
                if group >= Int(n_match_groups):
                    return None
        elif op == OP_RANGE:
            if arg < 0 or arg >= Int(n_count_leaves):
                return None
        elif op == OP_SOME:
            if arg < 0:
                return None
    for g in range(Int(total_match_refs)):
        var rule = Int(match_group_rules[unsafe_offset=g])
        if rule < 0 or rule >= Int(n_rules):
            return None

    var rs = unsafe_alloc[RuleSet](1)
    rs[] = RuleSet(
        n_rules,
        total_nodes,
        total_children,
        n_count_leaves,
        n_match_groups,
        total_match_refs,
        max_rule_nodes,
        _copy_i64(rule_node_offsets, n_rules + 1),
        _copy_i32(node_op, total_nodes),
        _copy_i64(node_child_offsets, total_nodes + 1),
        _copy_i32(node_children, total_children),
        _copy_i32(node_arg, total_nodes),
        _copy_u64(node_range_min, total_nodes),
        _copy_u64(node_range_max, total_nodes),
        _copy_i64(match_group_offsets, n_match_groups + 1),
        _copy_i32(match_group_rules, total_match_refs),
        unsafe_alloc[UInt64](Int(max_rule_nodes)),
    )
    return rs.unsafe_bitcast[UInt8]()


@export
def capamojo_eval(
    handle: Handle,
    n_scopes: Int64,
    leaf_counts: I32Ptr,
    leaf_present_words: U64Ptr,
    out_rule_words: U64Ptr,
) abi("C") -> Int32:
    """Evaluate every rule against up to 64 scopes, one bit per scope.

    `leaf_counts` is leaf-major [n_slots x n_scopes] int32 (location counts);
    `leaf_present_words` holds one uint64 per slot (feature-key presence).
    `out_rule_words` must hold n_rules uint64 slots. Returns 0 on success,
    1 on a NULL handle, 2 on an out-of-range scope count.
    """
    if not handle:
        return 1
    if n_scopes <= 0 or n_scopes > 64:
        return 2
    var rs = handle.value().unsafe_bitcast[RuleSet]()
    var n_rules = Int(rs[].n_rules)
    var lanes = Int(n_scopes)
    var lane_mask = UInt64(0xFFFF_FFFF_FFFF_FFFF)
    if lanes < 64:
        lane_mask = (UInt64(1) << UInt64(lanes)) - UInt64(1)

    var n_count_leaves = Int(rs[].n_count_leaves)
    var node_op = rs[].node_op
    var node_child_offsets = rs[].node_child_offsets
    var node_children = rs[].node_children
    var node_arg = rs[].node_arg
    var node_range_min = rs[].node_range_min
    var node_range_max = rs[].node_range_max
    var match_group_offsets = rs[].match_group_offsets
    var match_group_rules = rs[].match_group_rules
    var scratch = rs[].scratch

    for r in range(n_rules):
        var node_start = Int(rs[].rule_node_offsets[unsafe_offset=r])
        var node_end = Int(rs[].rule_node_offsets[unsafe_offset=r + 1])
        for i in range(node_start, node_end):
            var local = i - node_start
            var op = node_op[unsafe_offset=i]
            var c0 = Int(node_child_offsets[unsafe_offset=i])
            var c1 = Int(node_child_offsets[unsafe_offset=i + 1])
            var word = UInt64(0)
            if op == OP_AND:
                word = lane_mask
                for c in range(c0, c1):
                    var child = Int(node_children[unsafe_offset=c]) - node_start
                    word &= scratch[unsafe_offset=child]
            elif op == OP_OR:
                word = UInt64(0)
                for c in range(c0, c1):
                    var child = Int(node_children[unsafe_offset=c]) - node_start
                    word |= scratch[unsafe_offset=child]
            elif op == OP_NOT:
                var child = Int(node_children[unsafe_offset=c0]) - node_start
                word = ~scratch[unsafe_offset=child] & lane_mask
            elif op == OP_SOME:
                var threshold = Int(node_arg[unsafe_offset=i])
                if threshold <= 0:
                    # reference short-circuit semantics: at-least-zero with
                    # children is always true; with NO children it is false.
                    if c1 > c0:
                        word = lane_mask
                    else:
                        word = UInt64(0)
                else:
                    for lane in range(lanes):
                        var bit = UInt64(1) << UInt64(lane)
                        var hits = 0
                        for c in range(c0, c1):
                            var child = Int(node_children[unsafe_offset=c]) - node_start
                            if scratch[unsafe_offset=child] & bit != 0:
                                hits += 1
                        if hits >= threshold:
                            word |= bit
            elif op == OP_RANGE:
                var leaf = Int(node_arg[unsafe_offset=i])
                var lo = node_range_min[unsafe_offset=i]
                var hi = node_range_max[unsafe_offset=i]
                var base = leaf * lanes
                for lane in range(lanes):
                    var count = UInt64(
                        Int(leaf_counts[unsafe_offset=base + lane])
                    )
                    if lo <= count and count <= hi:
                        word |= UInt64(1) << UInt64(lane)
            else:  # OP_LEAF
                var arg = Int(node_arg[unsafe_offset=i])
                word = leaf_present_words[unsafe_offset=arg]
                if arg >= n_count_leaves:
                    var group = arg - n_count_leaves
                    var g0 = Int(match_group_offsets[unsafe_offset=group])
                    var g1 = Int(match_group_offsets[unsafe_offset=group + 1])
                    for g in range(g0, g1):
                        var ref_rule = Int(match_group_rules[unsafe_offset=g])
                        word |= out_rule_words[unsafe_offset=ref_rule]
            scratch[unsafe_offset=local] = word
        out_rule_words[unsafe_offset=r] = scratch[
            unsafe_offset=node_end - node_start - 1
        ] & lane_mask
    return 0


@export
def capamojo_ruleset_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var rs = handle.value().unsafe_bitcast[RuleSet]()
    rs[].rule_node_offsets.unsafe_free()
    rs[].node_op.unsafe_free()
    rs[].node_child_offsets.unsafe_free()
    rs[].node_children.unsafe_free()
    rs[].node_arg.unsafe_free()
    rs[].node_range_min.unsafe_free()
    rs[].node_range_max.unsafe_free()
    rs[].match_group_offsets.unsafe_free()
    rs[].match_group_rules.unsafe_free()
    rs[].scratch.unsafe_free()
    rs.unsafe_free()

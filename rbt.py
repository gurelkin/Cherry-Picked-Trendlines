from typing import Optional


class COLOR:
    """Defines color constants for Red-Black Tree nodes."""
    RED = "RED"
    BLACK = "BLACK"


class Node:
    """
    Represents a node in the Red-Black Tree with order statistics.

    @param key: The key associated with the node.
    @param color: The color of the node (RED or BLACK).
    @param left: Reference to the left child node.
    @param right: Reference to the right child node.
    @param parent: Reference to the parent node.
    @param size: The size of the subtree rooted at this node.
    """
    def __init__(
        self, 
        key: Optional[int] = None, 
        color: Optional[str] = None, 
        left: Optional["Node"] = None,
        right: Optional["Node"] = None, 
        parent: Optional["Node"] = None, 
        size: int = 1
    ):
        self.key = key
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent
        self.size = size


class RedBlackTree:
    """
    Implements a Red-Black Tree with order statistics.

    Supports insertion, deletion, rank selection, and size tracking.
    """
    def __init__(self):
        """Initializes an empty Red-Black Tree with a NIL sentinel node."""
        self.NIL = Node(key="NIL", color=COLOR.BLACK, size=0)
        self.T = self.NIL  # Root of the tree

    def select(self, x: Node, i: int) -> Node:
        """
        Returns the i-th smallest element in the subtree rooted at x.

        @param x: The root node of the subtree.
        @param i: The rank of the element to find.
        @return: The node corresponding to the i-th smallest element.
        """
        r = x.left.size + 1
        if i == r:
            return x
        elif i < r:
            return self.select(x.left, i)
        else:
            return self.select(x.right, i - r)

    def rank(self, x: Node) -> int:
        """
        Returns the rank of node x within the tree.

        @param x: The node whose rank is to be determined.
        @return: The rank of x.
        """
        r = x.left.size + 1
        y = x
        while y != self.T:
            if y == y.parent.right:
                r += y.parent.left.size + 1
            y = y.parent
        return r

    def find(self, x: Node, i: int) -> Node:
        """
        Searches for a node with key i in the subtree rooted at x.

        @param x: The root of the subtree to search.
        @param i: The key value to find.
        @return: The node with key i, or NIL if not found.
        """
        if x == self.NIL or x.key == i:
            return x
        if i < x.key:
            return self.find(x.left, i)
        return self.find(x.right, i)

    def insert(self, i: int) -> int:
        """
        Inserts a new key into the Red-Black Tree.

        @param i: The key value to insert.
        @return: The inserted key value.
        """
        if self.NIL != self.find(self.T, i):
            return 0

        z = Node(key=i)
        y = self.NIL
        x = self.T
        while x != self.NIL:
            y = x
            y.size += 1  # Increment size of each traversed node
            if z.key < x.key:
                x = x.left
            else:
                x = x.right

        z.parent = y
        if y == self.NIL:
            self.T = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

        z.left = z.right = self.NIL
        z.color = COLOR.RED
        self.insert_fixup(z)

        return i

    def insert_fixup(self, z: Node):
        """
        Restores Red-Black Tree properties after insertion.

        @param z: The newly inserted node.
        """
        while z.parent != self.NIL and z.parent.color == COLOR.RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y != self.NIL and y.color == COLOR.RED:
                    z.parent.color = COLOR.BLACK
                    y.color = COLOR.BLACK
                    z.parent.parent.color = COLOR.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self.left_rotate(z)
                    z.parent.color = COLOR.BLACK
                    z.parent.parent.color = COLOR.RED
                    self.right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == COLOR.RED:
                    z.parent.color = COLOR.BLACK
                    y.color = COLOR.BLACK
                    z.parent.parent.color = COLOR.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self.right_rotate(z)
                    z.parent.color = COLOR.BLACK
                    z.parent.parent.color = COLOR.RED
                    self.left_rotate(z.parent.parent)
        self.T.color = COLOR.BLACK

    def left_rotate(self, x: Node):
        """
        Performs a left rotation around node x.

        @param x: The node to rotate around.
        """
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.T = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def right_rotate(self, x: Node):
        """
        Performs a right rotation around node x.

        @param x: The node to rotate around.
        """
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.T = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def delete(self, i: int) -> int:
        """
        Deletes a node with key i from the Red-Black Tree.

        @param i: The key value to delete.
        @return: The deleted key value, or 0 if not found.
        """
        z = self.find(self.T, i)
        if z == self.NIL:
            return 0

        y = z if z.left == self.NIL or z.right == self.NIL else self.tree_successor(z)
        x = y.left if y.left != self.NIL else y.right

        x.parent = y.parent
        if y.parent == self.NIL:
            self.T = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x

        if y != z:
            z.key = y.key

        backprop = y.parent
        while backprop != self.NIL:
            backprop.size -= 1
            backprop = backprop.parent

        if y.color == COLOR.BLACK:
            self.delete_fixup(x)

        return i

    def size(self) -> int:
        """Returns the size of the tree (number of elements)."""
        return self.T.size

    def count_smaller_than(self, i: int) -> int:
        """
        Counts the number of elements in the tree smaller than i.

        @param i: The key value to compare.
        @return: The count of elements smaller than i.
        """
        x = self.T
        count = 0
        while x != self.NIL:
            if i < x.key:
                x = x.left
            elif i == x.key:
                count += x.left.size + 1
                break
            else:
                count += x.left.size + 1 # TODO: isn't that suppose to be `x.right`?
                x = x.right
        return count

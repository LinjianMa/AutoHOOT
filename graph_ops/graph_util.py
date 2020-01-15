import attr


@attr.s(eq=False)
class PseudoNode(object):
    node = attr.ib()
    literals = attr.ib(default=[])
    subscript = attr.ib(default='')

    def generate_subscript(self, uf):
        self.subscript = "".join(
            [uf.rootval(literal_name) for literal_name in self.literals])

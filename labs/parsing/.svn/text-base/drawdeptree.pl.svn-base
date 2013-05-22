#!/usr/bin/perl -w

# "drawdeptree.pl" is a visualization tool for dependency trees.
# Copyright (C) 2010  Terry Koo
#
# drawdeptree is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# drawdeptree is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with drawdeptree.  If not, see <http://www.gnu.org/licenses/>.



if(scalar(@ARGV) == 0) {
    print STDERR <<END;
usage: $0 [-gv] <conll-file> <compare-file>
    Draws dependency trees for the sentences in the file, assuming a
    CoNLL- or Malt-style format.  Specifically, assumes that sentences
    are given as one word per line with the last two columns being
    <head-index>,<label>; end-of-sentence is indicated by intervening
    blank lines.

    Output is written to <conll-style-file>.treeNNN.fig, where NNN is
    a 0-padded three-digit sentence index (1-origin).  The files are
    produced in Xfig format; use fig2dev to get what you want out of
    this.

    Option '-gv' indicates that trees will be immediately fig2dev-ed
    into eps and opened with gv.

END
    exit 1;
}

my $GV_FNAME = "";
my $GV_MODE = 0;
my $GV_GEOM = "";
my $COL_FTAG = 0;
my $GREP_MODE = 0;
my $GREP_REX = "";
my $SKIP_MODE = 0;
while(1) { # consume options
    if($ARGV[0] eq '-gv') {
        $GV_MODE = 1;
        $GV_GEOM = "+0+0";
    } elsif($ARGV[0] =~ m/^-ftag=(\d+)$/o) {
        $COL_FTAG = $1;
    } elsif($ARGV[0] =~ m/^-skip=(\d+)$/o) {
        $SKIP_MODE = $1;
    } elsif($ARGV[0] =~ m/^-grep=(.+)$/o) {
        $GREP_MODE = 1;
        $GREP_REX = $1;
    } else { last; }
    shift;
}

my $fname = shift;
my $cmpfn;
my $CMP_MODE = 0;
if(scalar(@ARGV) > 0) { $cmpfn = shift; $CMP_MODE = 1; }

my $PREAMBLE = <<END;
#FIG 3.2  Produced by xfig version 3.2.5
Landscape
Center
Metric
Letter
100.00
Single
-3
# Produced by drawdeptree.pl, written by Terry Koo
1200 2
END

my $FIG_VERSION = "3.2";


#     -1 = Default
#      0 = Black
#      1 = Blue
#      2 = Green
#      3 = Cyan
#      4 = Red
#      5 = Magenta
#      6 = Yellow
#      7 = White
#   8-11 = four shades of blue (dark to lighter)
#  12-14 = three shades of green (dark to lighter)
#  15-17 = three shades of cyan (dark to lighter)
#  18-20 = three shades of red (dark to lighter)
#  21-23 = three shades of magenta (dark to lighter)
#  24-26 = three shades of brown (dark to lighter)
#  27-30 = four shades of pink (dark to lighter)
#     31 = Gold
my $COLOR = 0;
my $COLOR_ERR = 4;
my $COLOR_COR = 1;

# -1 = Default
#  0 = Solid
#  1 = Dashed
#  2 = Dotted
#  3 = Dash-dotted
#  4 = Dash-double-dotted
#  5 = Dash-triple-dotted
my $LINE_STYLE = 0;

# measured in 1/80 of an inch
my $LINE_THICK = 3;
#  0 = Miter (the default in xfig 2.1 and earlier)
#  1 = Round
#  2 = Bevel
my $JOIN_STYLE = 0;

#  0 = Butt (the default in xfig 2.1 and earlier)
#  1 = Round
#  2 = Projecting
my $CAP_STYLE = 0;

#  0 = Stick-type (the default in xfig 2.1 and earlier)
#  1 = Closed triangle:
#  2 = Closed with "indented" butt:
#  3 = Closed with "pointed" butt:
my $ARROW_TYPE = 1;

#  0 = Hollow (actually filled with white)
#  1 = Filled with pen_color
my $ARROW_STYLE = 0;

# line-thickness of arrowhead outline, measured in 1/80 inch
my $ARROW_THICK = '1.00';

# dimensions of arrowhead, measured in Fig units
my $ARROW_WIDTH  = '135.00';
my $ARROW_HEIGHT = '165.00';

# default depth
my $DEPTH = 50;

# the y-height at which all arcs begin and end.  mostly arbitrary
# since EPS processing will determine the BBox anyway.
my $BASELINE = 4000;



# 1 if arrows are drawn with head pointing to modifier, 0 otherwise
my $HEAD_TO_MOD = 1;
# opposite
my $MOD_TO_HEAD = 1 - $HEAD_TO_MOD;

# the default angle which all arcs have.  in the future, arcs may be
# allowed to have differing angles, computed for aesthetic or
# pragmatic (fitting things on screen) reasons.
my $ARC_ANGLE = 120;
use constant PI => 4 * atan2(1, 1);




# font: 0 = Times Roman
my $FONT_FACE = 0;

# font-flags: bit 2 = 1, use postscript fonts
my $FONT_FLAGS = 4;

# point-size of the font
my $FONT_POINTS = 12;

# em-size of the font in fig units.  1 point = 1/72 in, and 1 Fig unit
# is 1/1200 in
my $FONT_EM_SIZE = 1200*$FONT_POINTS/72;

# an approximation: given the em-size of a font, how wide (in
# em-units) is any given character on average?
my $EM_TO_CHAR_WIDTH = 1/3;




# column-index of the word
my $COL_WORD = 2;  # 2nd from beginning
my $COL_HEAD = -2; # 2nd from end

# read the file and convert its trees
open(TREES, ($fname =~ m/\.gz$/io ? "gunzip -c '$fname' |" : "< $fname"))
    or die "couldn't open tree-file '$fname'";
if($CMP_MODE) {
    open(CMPF, ($cmpfn =~ m/\.gz$/io ? "gunzip -c '$cmpfn' |" : "< $cmpfn"))
        or die "couldn't open tree-file '$cmpfn'";
}
my $sent = 0;
my $opened = 0;
my $INIT_X = 100;
my $x = -1000; # current x-position
my @xpos = ();
my @ftags = ();
my @words = ();
my @heads = ();
my @cheads = ();
while(<TREES>) {
    my @elts = split;
    if(scalar(@elts) == 0) {
        if($CMP_MODE) {
            my $cline = <CMPF>;
            die if(!defined($cline));
            my @celts = split(' ', $cline);
            die if(scalar(@celts) != 0);
        }
        my $print_this = 1;
        if($GREP_MODE) { # print only for matches
            $print_this = 0;
            foreach my $tok (@words, @ftags) {
                if($tok =~ m/\Q$GREP_REX\E/) {
                    $print_this = 1; last;
                }
            }
        }
        # optionally skip the first n sentences
        $print_this = 0 if($sent <= $SKIP_MODE);
        print OUT &tree(\@xpos, \@ftags, \@words, \@heads, \@cheads)
            if($print_this);
        close(OUT);
        $opened = 0;
        # reset state
        @xpos = ();
        @ftags = ();
        @words = ();
        @heads = ();
        @cheads = ();
        # process immediately if desired
        if($GV_MODE && $print_this) {
            my $epsf = $GV_FNAME;
            $epsf =~ s/\.fig$/.eps/;
            system("fig2dev -L eps -p dummy $GV_FNAME $epsf") == 0
                or die "fig2dev failed!";
            system("gv -g $GV_GEOM $epsf") == 0
                or die "gv failed!";
        }
        next;
    } else {
        if($opened == 0) {
            ++$sent;
            # select proper output file
            my $outfn = sprintf("%s.tree%03d.fig",
                                ($CMP_MODE ? $cmpfn :
                                 ($fname eq '-' ? 'stdin' : $fname)),
                                $sent);
            $GV_FNAME = $outfn if($GV_MODE);
            open(OUT, "> $outfn")
                or die "couldn't open out-file '$outfn'";
            $opened = 1;
            # save root token
            $x = $INIT_X;
            push(@xpos, $x);
            push(@ftags, " ");
            push(@words, " * ");
            push(@heads, -1);
            push(@cheads, -1);
            $x += 2*$FONT_EM_SIZE;
        }
    }
    # extract the word
    my $word = ($COL_WORD < 0
                ? $elts[$COL_WORD + scalar(@elts)]
                : $elts[$COL_WORD - 1]);
    # extract the head-index
    my $head = ($COL_HEAD < 0
                ? $elts[$COL_HEAD + scalar(@elts)]
                : $elts[$COL_HEAD - 1]);
    # add the ftag if specified
    my $ftag = " ";
    if($COL_FTAG != 0) {
        $ftag = ($COL_FTAG < 0
                 ? $elts[$COL_FTAG + scalar(@elts)]
                 : $elts[$COL_FTAG - 1]);
    }
    my $width = &textwidth($word);
    # advance x to center position
    $x += 0.5*$width;
    # save position, word, and head
    push(@xpos, $x);
    push(@ftags, $ftag);
    push(@words, $word);
    push(@heads, $head);
    if($CMP_MODE) {
        my $cline = <CMPF>;
        die if(!defined($cline));
        my @celts = split(' ', $cline);
        my $chead = ($COL_HEAD < 0
                     ? $celts[$COL_HEAD + scalar(@celts)]
                     : $celts[$COL_HEAD - 1]);
        push(@cheads, $chead);
    } else { push(@cheads, $head); }
    # advance x to past the current word
    $x += 0.5*$width + 2*$FONT_EM_SIZE;
}
close(TREES);
close(CMPF);

exit;



# construct a string representing a fig-file for the given tree
sub tree {
    my $xpos = shift;
    my $ftags = shift;
    my $words = shift;
    my $heads = shift;
    my $cheads = shift;

    my $n = scalar(@$xpos);
    die if($n != scalar(@$ftags));
    die if($n != scalar(@$words));
    die if($n != scalar(@$heads));
    die if($heads->[0] != -1);
    die if($cheads->[0] != -1);

    # initialize with preamble
    my $ret = "$PREAMBLE\n";

    # add the root
    $ret .= &text($xpos->[0], $words->[0], 0, 2*$FONT_POINTS);
    # add each word
    foreach my $i (1 .. ($n - 1)) {
        if($COL_FTAG == 0) {
            $ret .= &text($xpos->[$i], $words->[$i], 0);
        } else {
            $ret .= &text($xpos->[$i], $ftags->[$i], 0);
            $ret .= &text($xpos->[$i], $words->[$i], 1);
        }
    }

    # add dependencies; we also set depths so that dependencies higher
    # up in the tree are "on top of" dependencies lower in the tree
    # (higher and lower in terms of the hierarchy of the tree).  this
    # is mostly for aesthetic reasons; without this the arrowheads can
    # sometimes get buried beneath a pile of arrowtails.
    my @depths = map {-1} (1 .. $n);
    # compute the depths first
    foreach my $i (1 .. ($n - 1)) { &rdepth(\@depths, $heads, $i); }
    # now append the dependencies with their appropriate depths
    foreach my $i (1 .. ($n - 1)) {
        my $hidx = $heads->[$i];
        die if($hidx < 0);
        die if($hidx >= $n);
        die if($depths[$i] <= 0);
        if($CMP_MODE) {
            my $chidx = $cheads->[$i];
            if($chidx != $hidx) { # differing head
                my $d = $depths[$i];
                $ret .= &arc($xpos->[$hidx],  $xpos->[$i], $d, $COLOR_COR);
                $ret .= &arc($xpos->[$chidx], $xpos->[$i], $d, $COLOR_ERR);
            } else {
                $ret .= &arc($xpos->[$hidx], $xpos->[$i], $depths[$i]);
            }
        } else {
            $ret .= &arc($xpos->[$hidx], $xpos->[$i], $depths[$i]);
        }
    }

    return $ret;
}



# recursively compute depths
sub rdepth {
    my $depths = shift;
    my $heads = shift;
    my $i = shift;

    # depth already known
    return $depths->[$i] if($depths->[$i] >= 1);

    my $h = $heads->[$i];
    # a root dependency; base case
    return ($depths->[$i] = 1) if($h == 0);

    # general case
    return ($depths->[$i] = &rdepth($depths, $heads, $h) + 1);
}



# compute a best guess at the width of the given text
sub textwidth {
    my $char_width = $EM_TO_CHAR_WIDTH*$FONT_EM_SIZE;
    my $str = shift;
    return length($str)*$char_width;
}


# obtain a string that creates a text item centered at $x and
# positioned "sufficiently" below $BASELINE
sub text {
    my $x = shift;
    my $str = shift;
    my $below = shift;
    my $points = shift;
    $points = $FONT_POINTS if(!defined($points));

    my $y = $BASELINE + ($below + 1)*$FONT_EM_SIZE +
        0.5*(($points - $FONT_POINTS)/$FONT_POINTS)*$FONT_EM_SIZE;
    my $width = &textwidth($str);

    # round to integers
    $x = int(0.5 + $x);
    $y = int(0.5 + $y);

    # construct and return a string
    return

# type   name        (brief description)
# ----   ----        -------------------
# int    object      (always 4)
        "4".
# int    sub_type    (0: Left justified
#                     1: Center justified
#                     2: Right justified)
        " 1".
# int    color       (enumeration type)
        " $COLOR".
# int    depth       (enumeration type)
        " $DEPTH".
# int    pen_style   (enumeration , not used)
        " -1".
# int    font        (enumeration type)
        " $FONT_FACE".
# float  font_size   (font size in points)
        " $points".
# float  angle       (radians, the angle of the text)
        " 0.000".
# int    font_flags  (bit vector)
        " $FONT_FLAGS".
# float  height      (Fig units)
        " $FONT_EM_SIZE".
# float  length      (Fig units)
        " $width".
# int    x, y        (Fig units, coordinate of the origin
#                     of the string.  If sub_type = 0, it is
#                     the lower left corner of the string.
#                     If sub_type = 1, it is the lower
#                     center.  Otherwise it is the lower
#                     right corner of the string.)
        " $x $y".
# char   string[]    (ASCII characters; starts after a blank
#                     character following the last number and
#                     ends before the sequence '\001'.  This
#                     sequence is not part of the string.
#                     Characters above octal 177 are
#                     represented by \xxx where xxx is the
#                     octal value.  This permits fig files to
#                     be edited with 7-bit editors and sent
#                     by e-mail without data loss.
#                     Note that the string may contain '\n'.)
        " $str\\001".

        # END OF TEXT
        "\n";


#### SETTINGS ABOVE ##################################################
# The font_flags field is defined as follows:
# Bit    Description
#
# 0      Rigid text (text doesn't scale when scaling compound objects)
# 1      Special text (for LaTeX)
# 2      PostScript font (otherwise LaTeX font is used)
# 3      Hidden text
#
# The font field is defined as follows:
#
# For font_flags bit 2 = 0 (LaTeX fonts):
# 0  Default font
# 1  Roman
# 2  Bold
# 3  Italic
# 4  Sans Serif
# 5  Typewriter
#
# For font_flags bit 2 = 1 (PostScript fonts):
# -1  Default font
#  0  Times Roman
#  1  Times Italic
#  2  Times Bold
#  3  Times Bold Italic
#  4  AvantGarde Book
#  5  AvantGarde Book Oblique
#  6  AvantGarde Demi
#  7  AvantGarde Demi Oblique
#  8  Bookman Light
#  9  Bookman Light Italic
# 10  Bookman Demi
# 11  Bookman Demi Italic
# 12  Courier
# 13  Courier Oblique
# 14  Courier Bold
# 15  Courier Bold Oblique
# 16  Helvetica
# 17  Helvetica Oblique
# 18  Helvetica Bold
# 19  Helvetica Bold Oblique
# 20  Helvetica Narrow
# 21  Helvetica Narrow Oblique
# 22  Helvetica Narrow Bold
# 23  Helvetica Narrow Bold Oblique
# 24  New Century Schoolbook Roman
# 25  New Century Schoolbook Italic
# 26  New Century Schoolbook Bold
# 27  New Century Schoolbook Bold Italic
# 28  Palatino Roman
# 29  Palatino Italic
# 30  Palatino Bold
# 31  Palatino Bold Italic
# 32  Symbol
# 33  Zapf Chancery Medium Italic
# 34  Zapf Dingbats
}





# obtain a string that creates an arc extending from ($h, $BASELINE)
# to ($m, $BASELINE)
sub arc {
    my $h = shift; # head position
    my $m = shift; # mod position
    my $d = shift; # depth within figure
    my $c = shift; # color of arc
    $c = $COLOR if(!defined($c));
    my $width = ($h < $m ? $m - $h : $h - $m);
    die if($width <= 0);

    # angle varies with depth
    my $full_rads = PI*$ARC_ANGLE/180;
    my $rads = (3/4)*$full_rads + (1/4)*$full_rads/sqrt($d);
    # static angle
#     my $rads = PI*$ARC_ANGLE/180;

    # trig at work!  we compute here the height of the midpoint of the arc
    # above the baseline, as well as the depth of the center of the arc's
    # circle below the baseline.
    my $rad_half = 0.5*$rads;
    my $tan_half = sin($rad_half)/cos($rad_half); # no tan() func
    my $wid_half = 0.5*$width;
    my $depth  = $wid_half/$tan_half;
    my $radius = $wid_half/sin($rad_half);
    my $height = $radius - $depth;
    die if($depth <= 0);
    die if($radius <= 0);
    die if($height <= 0);

    # the center of the arc
    my $center_x = 0.5*($h + $m);
    my $center_y = $BASELINE + $depth; # NB: screen-style axes
    # the point at the top of the arc
    my $arctop_x = $center_x;
    my $arctop_y = $BASELINE - $height; # NB: screen-style axes

    # round to integers
    $arctop_x = int(0.5 + $arctop_x);
    $arctop_y = int(0.5 + $arctop_y);
    $h = int(0.5 + $h);
    $m = int(0.5 + $m);

    # construct and return arc-arrow string
    return

# First line:
#   type   name                (brief description)
#   ----   ----                -------------------
#   int    object_code         (always 5)
        "5".
#   int    sub_type            (1: open ended arc
#                               2: pie-wedge (closed))
        " 1".
#   int    line_style          (enumeration type)
        " $LINE_STYLE".
#   int    line_thickness      (1/80 inch)
        " $LINE_THICK".
#   int    pen_color           (enumeration type, pen color)
        " $c".
#   int    fill_color          (enumeration type, fill color)
        " 7". # (white, but won't be used anyway)
#   int    depth               (enumeration type)
        " $d".
#   int    pen_style           (pen style, not used)
        " -1".
#   int    area_fill           (enumeration type, -1 = no fill)
        " -1".
#   float  style_val           (1/80 inch)
        " 0.000".
#   int    cap_style           (enumeration type)
        " $CAP_STYLE".
#   int    direction           (0: clockwise, 1: counterclockwise)
        " ".($h < $m ? "0" : "1"). # left-headed = clockwise
#   int    forward_arrow       (0: no forward arrow, 1: on)
        " $HEAD_TO_MOD".
#   int    backward_arrow      (0: no backward arrow, 1: on)
        " $MOD_TO_HEAD".
#   float  center_x, center_y  (center of the arc)
        " $center_x $center_y".
#   int    x1, y1              (Fig units, the 1st point the user entered)
        " $h $BASELINE".
#   int    x2, y2              (Fig units, the 2nd point)
        " $arctop_x $arctop_y".
#   int    x3, y3              (Fig units, the last point)
        " $m $BASELINE".

        # NEXT LINE
        "\n       ".

# Forward arrow line (Optional; absent if forward_arrow is 0):
#   type   name                (brief description)
#   ----   ----                -------------------
#   int    arrow_type          (enumeration type)
        " $ARROW_TYPE".
#   int    arrow_style         (enumeration type)
        " $ARROW_STYLE".
#   float  arrow_thickness     (1/80 inch)
        " $ARROW_THICK".
#   float  arrow_width         (Fig units)
        " $ARROW_WIDTH".
#   float  arrow_height        (Fig units)
        " $ARROW_HEIGHT".

        # END OF ARC
        "\n";


#### UNUSED #####################################################
# Backward arrow line (Optional; absent if backward_arrow is 0):
#   type   name                (brief description)
#   ----   ----                -------------------
#   int    arrow_type          (enumeration type)
#   int    arrow_style         (enumeration type)
#   float  arrow_thickness     (1/80 inch)
#   float  arrow_width         (Fig units)
#   float  arrow_height        (Fig units)
}

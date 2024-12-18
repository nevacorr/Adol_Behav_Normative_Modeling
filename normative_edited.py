#!/opt/conda/bin/python
#######
# This code is adapted from the normative.py module of the the Predictive Clinical Neuroscience Toolkit
# which is available at https://pcntoolkit.readthedocs.io/en/latest/. The only modification from the original
# code is the commenting out of lines 1569 and 1570.
# The license associated with the PCN toolkit is copied below. The code is below the license.
#######
#                    GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
#
#                             Preamble
#
#   The GNU General Public License is a free, copyleft license for
# software and other kinds of works.
#
#   The licenses for most software and other practical works are designed
# to take away your freedom to share and change the works.  By contrast,
# the GNU General Public License is intended to guarantee your freedom to
# share and change all versions of a program--to make sure it remains free
# software for all its users.  We, the Free Software Foundation, use the
# GNU General Public License for most of our software; it applies also to
# any other work released this way by its authors.  You can apply it to
# your programs, too.
#
#   When we speak of free software, we are referring to freedom, not
# price.  Our General Public Licenses are designed to make sure that you
# have the freedom to distribute copies of free software (and charge for
# them if you wish), that you receive source code or can get it if you
# want it, that you can change the software or use pieces of it in new
# free programs, and that you know you can do these things.
#
#   To protect your rights, we need to prevent others from denying you
# these rights or asking you to surrender the rights.  Therefore, you have
# certain responsibilities if you distribute copies of the software, or if
# you modify it: responsibilities to respect the freedom of others.
#
#   For example, if you distribute copies of such a program, whether
# gratis or for a fee, you must pass on to the recipients the same
# freedoms that you received.  You must make sure that they, too, receive
# or can get the source code.  And you must show them these terms so they
# know their rights.
#
#   Developers that use the GNU GPL protect your rights with two steps:
# (1) assert copyright on the software, and (2) offer you this License
# giving you legal permission to copy, distribute and/or modify it.
#
#   For the developers' and authors' protection, the GPL clearly explains
# that there is no warranty for this free software.  For both users' and
# authors' sake, the GPL requires that modified versions be marked as
# changed, so that their problems will not be attributed erroneously to
# authors of previous versions.
#
#   Some devices are designed to deny users access to install or run
# modified versions of the software inside them, although the manufacturer
# can do so.  This is fundamentally incompatible with the aim of
# protecting users' freedom to change the software.  The systematic
# pattern of such abuse occurs in the area of products for individuals to
# use, which is precisely where it is most unacceptable.  Therefore, we
# have designed this version of the GPL to prohibit the practice for those
# products.  If such problems arise substantially in other domains, we
# stand ready to extend this provision to those domains in future versions
# of the GPL, as needed to protect the freedom of users.
#
#   Finally, every program is threatened constantly by software patents.
# States should not allow patents to restrict development and use of
# software on general-purpose computers, but in those that do, we wish to
# avoid the special danger that patents applied to a free program could
# make it effectively proprietary.  To prevent this, the GPL assures that
# patents cannot be used to render the program non-free.
#
#   The precise terms and conditions for copying, distribution and
# modification follow.
#
#                        TERMS AND CONDITIONS
#
#   0. Definitions.
#
#   "This License" refers to version 3 of the GNU General Public License.
#
#   "Copyright" also means copyright-like laws that apply to other kinds of
# works, such as semiconductor masks.
#
#   "The Program" refers to any copyrightable work licensed under this
# License.  Each licensee is addressed as "you".  "Licensees" and
# "recipients" may be individuals or organizations.
#
#   To "modify" a work means to copy from or adapt all or part of the work
# in a fashion requiring copyright permission, other than the making of an
# exact copy.  The resulting work is called a "modified version" of the
# earlier work or a work "based on" the earlier work.
#
#   A "covered work" means either the unmodified Program or a work based
# on the Program.
#
#   To "propagate" a work means to do anything with it that, without
# permission, would make you directly or secondarily liable for
# infringement under applicable copyright law, except executing it on a
# computer or modifying a private copy.  Propagation includes copying,
# distribution (with or without modification), making available to the
# public, and in some countries other activities as well.
#
#   To "convey" a work means any kind of propagation that enables other
# parties to make or receive copies.  Mere interaction with a user through
# a computer network, with no transfer of a copy, is not conveying.
#
#   An interactive user interface displays "Appropriate Legal Notices"
# to the extent that it includes a convenient and prominently visible
# feature that (1) displays an appropriate copyright notice, and (2)
# tells the user that there is no warranty for the work (except to the
# extent that warranties are provided), that licensees may convey the
# work under this License, and how to view a copy of this License.  If
# the interface presents a list of user commands or options, such as a
# menu, a prominent item in the list meets this criterion.
#
#   1. Source Code.
#
#   The "source code" for a work means the preferred form of the work
# for making modifications to it.  "Object code" means any non-source
# form of a work.
#
#   A "Standard Interface" means an interface that either is an official
# standard defined by a recognized standards body, or, in the case of
# interfaces specified for a particular programming language, one that
# is widely used among developers working in that language.
#
#   The "System Libraries" of an executable work include anything, other
# than the work as a whole, that (a) is included in the normal form of
# packaging a Major Component, but which is not part of that Major
# Component, and (b) serves only to enable use of the work with that
# Major Component, or to implement a Standard Interface for which an
# implementation is available to the public in source code form.  A
# "Major Component", in this context, means a major essential component
# (kernel, window system, and so on) of the specific operating system
# (if any) on which the executable work runs, or a compiler used to
# produce the work, or an object code interpreter used to run it.
#
#   The "Corresponding Source" for a work in object code form means all
# the source code needed to generate, install, and (for an executable
# work) run the object code and to modify the work, including scripts to
# control those activities.  However, it does not include the work's
# System Libraries, or general-purpose tools or generally available free
# programs which are used unmodified in performing those activities but
# which are not part of the work.  For example, Corresponding Source
# includes interface definition files associated with source files for
# the work, and the source code for shared libraries and dynamically
# linked subprograms that the work is specifically designed to require,
# such as by intimate data communication or control flow between those
# subprograms and other parts of the work.
#
#   The Corresponding Source need not include anything that users
# can regenerate automatically from other parts of the Corresponding
# Source.
#
#   The Corresponding Source for a work in source code form is that
# same work.
#
#   2. Basic Permissions.
#
#   All rights granted under this License are granted for the term of
# copyright on the Program, and are irrevocable provided the stated
# conditions are met.  This License explicitly affirms your unlimited
# permission to run the unmodified Program.  The output from running a
# covered work is covered by this License only if the output, given its
# content, constitutes a covered work.  This License acknowledges your
# rights of fair use or other equivalent, as provided by copyright law.
#
#   You may make, run and propagate covered works that you do not
# convey, without conditions so long as your license otherwise remains
# in force.  You may convey covered works to others for the sole purpose
# of having them make modifications exclusively for you, or provide you
# with facilities for running those works, provided that you comply with
# the terms of this License in conveying all material for which you do
# not control copyright.  Those thus making or running the covered works
# for you must do so exclusively on your behalf, under your direction
# and control, on terms that prohibit them from making any copies of
# your copyrighted material outside their relationship with you.
#
#   Conveying under any other circumstances is permitted solely under
# the conditions stated below.  Sublicensing is not allowed; section 10
# makes it unnecessary.
#
#   3. Protecting Users' Legal Rights From Anti-Circumvention Law.
#
#   No covered work shall be deemed part of an effective technological
# measure under any applicable law fulfilling obligations under article
# 11 of the WIPO copyright treaty adopted on 20 December 1996, or
# similar laws prohibiting or restricting circumvention of such
# measures.
#
#   When you convey a covered work, you waive any legal power to forbid
# circumvention of technological measures to the extent such circumvention
# is effected by exercising rights under this License with respect to
# the covered work, and you disclaim any intention to limit operation or
# modification of the work as a means of enforcing, against the work's
# users, your or third parties' legal rights to forbid circumvention of
# technological measures.
#
#   4. Conveying Verbatim Copies.
#
#   You may convey verbatim copies of the Program's source code as you
# receive it, in any medium, provided that you conspicuously and
# appropriately publish on each copy an appropriate copyright notice;
# keep intact all notices stating that this License and any
# non-permissive terms added in accord with section 7 apply to the code;
# keep intact all notices of the absence of any warranty; and give all
# recipients a copy of this License along with the Program.
#
#   You may charge any price or no price for each copy that you convey,
# and you may offer support or warranty protection for a fee.
#
#   5. Conveying Modified Source Versions.
#
#   You may convey a work based on the Program, or the modifications to
# produce it from the Program, in the form of source code under the
# terms of section 4, provided that you also meet all of these conditions:
#
#     a) The work must carry prominent notices stating that you modified
#     it, and giving a relevant date.
#
#     b) The work must carry prominent notices stating that it is
#     released under this License and any conditions added under section
#     7.  This requirement modifies the requirement in section 4 to
#     "keep intact all notices".
#
#     c) You must license the entire work, as a whole, under this
#     License to anyone who comes into possession of a copy.  This
#     License will therefore apply, along with any applicable section 7
#     additional terms, to the whole of the work, and all its parts,
#     regardless of how they are packaged.  This License gives no
#     permission to license the work in any other way, but it does not
#     invalidate such permission if you have separately received it.
#
#     d) If the work has interactive user interfaces, each must display
#     Appropriate Legal Notices; however, if the Program has interactive
#     interfaces that do not display Appropriate Legal Notices, your
#     work need not make them do so.
#
#   A compilation of a covered work with other separate and independent
# works, which are not by their nature extensions of the covered work,
# and which are not combined with it such as to form a larger program,
# in or on a volume of a storage or distribution medium, is called an
# "aggregate" if the compilation and its resulting copyright are not
# used to limit the access or legal rights of the compilation's users
# beyond what the individual works permit.  Inclusion of a covered work
# in an aggregate does not cause this License to apply to the other
# parts of the aggregate.
#
#   6. Conveying Non-Source Forms.
#
#   You may convey a covered work in object code form under the terms
# of sections 4 and 5, provided that you also convey the
# machine-readable Corresponding Source under the terms of this License,
# in one of these ways:
#
#     a) Convey the object code in, or embodied in, a physical product
#     (including a physical distribution medium), accompanied by the
#     Corresponding Source fixed on a durable physical medium
#     customarily used for software interchange.
#
#     b) Convey the object code in, or embodied in, a physical product
#     (including a physical distribution medium), accompanied by a
#     written offer, valid for at least three years and valid for as
#     long as you offer spare parts or customer support for that product
#     model, to give anyone who possesses the object code either (1) a
#     copy of the Corresponding Source for all the software in the
#     product that is covered by this License, on a durable physical
#     medium customarily used for software interchange, for a price no
#     more than your reasonable cost of physically performing this
#     conveying of source, or (2) access to copy the
#     Corresponding Source from a network server at no charge.
#
#     c) Convey individual copies of the object code with a copy of the
#     written offer to provide the Corresponding Source.  This
#     alternative is allowed only occasionally and noncommercially, and
#     only if you received the object code with such an offer, in accord
#     with subsection 6b.
#
#     d) Convey the object code by offering access from a designated
#     place (gratis or for a charge), and offer equivalent access to the
#     Corresponding Source in the same way through the same place at no
#     further charge.  You need not require recipients to copy the
#     Corresponding Source along with the object code.  If the place to
#     copy the object code is a network server, the Corresponding Source
#     may be on a different server (operated by you or a third party)
#     that supports equivalent copying facilities, provided you maintain
#     clear directions next to the object code saying where to find the
#     Corresponding Source.  Regardless of what server hosts the
#     Corresponding Source, you remain obligated to ensure that it is
#     available for as long as needed to satisfy these requirements.
#
#     e) Convey the object code using peer-to-peer transmission, provided
#     you inform other peers where the object code and Corresponding
#     Source of the work are being offered to the general public at no
#     charge under subsection 6d.
#
#   A separable portion of the object code, whose source code is excluded
# from the Corresponding Source as a System Library, need not be
# included in conveying the object code work.
#
#   A "User Product" is either (1) a "consumer product", which means any
# tangible personal property which is normally used for personal, family,
# or household purposes, or (2) anything designed or sold for incorporation
# into a dwelling.  In determining whether a product is a consumer product,
# doubtful cases shall be resolved in favor of coverage.  For a particular
# product received by a particular user, "normally used" refers to a
# typical or common use of that class of product, regardless of the status
# of the particular user or of the way in which the particular user
# actually uses, or expects or is expected to use, the product.  A product
# is a consumer product regardless of whether the product has substantial
# commercial, industrial or non-consumer uses, unless such uses represent
# the only significant mode of use of the product.
#
#   "Installation Information" for a User Product means any methods,
# procedures, authorization keys, or other information required to install
# and execute modified versions of a covered work in that User Product from
# a modified version of its Corresponding Source.  The information must
# suffice to ensure that the continued functioning of the modified object
# code is in no case prevented or interfered with solely because
# modification has been made.
#
#   If you convey an object code work under this section in, or with, or
# specifically for use in, a User Product, and the conveying occurs as
# part of a transaction in which the right of possession and use of the
# User Product is transferred to the recipient in perpetuity or for a
# fixed term (regardless of how the transaction is characterized), the
# Corresponding Source conveyed under this section must be accompanied
# by the Installation Information.  But this requirement does not apply
# if neither you nor any third party retains the ability to install
# modified object code on the User Product (for example, the work has
# been installed in ROM).
#
#   The requirement to provide Installation Information does not include a
# requirement to continue to provide support service, warranty, or updates
# for a work that has been modified or installed by the recipient, or for
# the User Product in which it has been modified or installed.  Access to a
# network may be denied when the modification itself materially and
# adversely affects the operation of the network or violates the rules and
# protocols for communication across the network.
#
#   Corresponding Source conveyed, and Installation Information provided,
# in accord with this section must be in a format that is publicly
# documented (and with an implementation available to the public in
# source code form), and must require no special password or key for
# unpacking, reading or copying.
#
#   7. Additional Terms.
#
#   "Additional permissions" are terms that supplement the terms of this
# License by making exceptions from one or more of its conditions.
# Additional permissions that are applicable to the entire Program shall
# be treated as though they were included in this License, to the extent
# that they are valid under applicable law.  If additional permissions
# apply only to part of the Program, that part may be used separately
# under those permissions, but the entire Program remains governed by
# this License without regard to the additional permissions.
#
#   When you convey a copy of a covered work, you may at your option
# remove any additional permissions from that copy, or from any part of
# it.  (Additional permissions may be written to require their own
# removal in certain cases when you modify the work.)  You may place
# additional permissions on material, added by you to a covered work,
# for which you have or can give appropriate copyright permission.
#
#   Notwithstanding any other provision of this License, for material you
# add to a covered work, you may (if authorized by the copyright holders of
# that material) supplement the terms of this License with terms:
#
#     a) Disclaiming warranty or limiting liability differently from the
#     terms of sections 15 and 16 of this License; or
#
#     b) Requiring preservation of specified reasonable legal notices or
#     author attributions in that material or in the Appropriate Legal
#     Notices displayed by works containing it; or
#
#     c) Prohibiting misrepresentation of the origin of that material, or
#     requiring that modified versions of such material be marked in
#     reasonable ways as different from the original version; or
#
#     d) Limiting the use for publicity purposes of names of licensors or
#     authors of the material; or
#
#     e) Declining to grant rights under trademark law for use of some
#     trade names, trademarks, or service marks; or
#
#     f) Requiring indemnification of licensors and authors of that
#     material by anyone who conveys the material (or modified versions of
#     it) with contractual assumptions of liability to the recipient, for
#     any liability that these contractual assumptions directly impose on
#     those licensors and authors.
#
#   All other non-permissive additional terms are considered "further
# restrictions" within the meaning of section 10.  If the Program as you
# received it, or any part of it, contains a notice stating that it is
# governed by this License along with a term that is a further
# restriction, you may remove that term.  If a license document contains
# a further restriction but permits relicensing or conveying under this
# License, you may add to a covered work material governed by the terms
# of that license document, provided that the further restriction does
# not survive such relicensing or conveying.
#
#   If you add terms to a covered work in accord with this section, you
# must place, in the relevant source files, a statement of the
# additional terms that apply to those files, or a notice indicating
# where to find the applicable terms.
#
#   Additional terms, permissive or non-permissive, may be stated in the
# form of a separately written license, or stated as exceptions;
# the above requirements apply either way.
#
#   8. Termination.
#
#   You may not propagate or modify a covered work except as expressly
# provided under this License.  Any attempt otherwise to propagate or
# modify it is void, and will automatically terminate your rights under
# this License (including any patent licenses granted under the third
# paragraph of section 11).
#
#   However, if you cease all violation of this License, then your
# license from a particular copyright holder is reinstated (a)
# provisionally, unless and until the copyright holder explicitly and
# finally terminates your license, and (b) permanently, if the copyright
# holder fails to notify you of the violation by some reasonable means
# prior to 60 days after the cessation.
#
#   Moreover, your license from a particular copyright holder is
# reinstated permanently if the copyright holder notifies you of the
# violation by some reasonable means, this is the first time you have
# received notice of violation of this License (for any work) from that
# copyright holder, and you cure the violation prior to 30 days after
# your receipt of the notice.
#
#   Termination of your rights under this section does not terminate the
# licenses of parties who have received copies or rights from you under
# this License.  If your rights have been terminated and not permanently
# reinstated, you do not qualify to receive new licenses for the same
# material under section 10.
#
#   9. Acceptance Not Required for Having Copies.
#
#   You are not required to accept this License in order to receive or
# run a copy of the Program.  Ancillary propagation of a covered work
# occurring solely as a consequence of using peer-to-peer transmission
# to receive a copy likewise does not require acceptance.  However,
# nothing other than this License grants you permission to propagate or
# modify any covered work.  These actions infringe copyright if you do
# not accept this License.  Therefore, by modifying or propagating a
# covered work, you indicate your acceptance of this License to do so.
#
#   10. Automatic Licensing of Downstream Recipients.
#
#   Each time you convey a covered work, the recipient automatically
# receives a license from the original licensors, to run, modify and
# propagate that work, subject to this License.  You are not responsible
# for enforcing compliance by third parties with this License.
#
#   An "entity transaction" is a transaction transferring control of an
# organization, or substantially all assets of one, or subdividing an
# organization, or merging organizations.  If propagation of a covered
# work results from an entity transaction, each party to that
# transaction who receives a copy of the work also receives whatever
# licenses to the work the party's predecessor in interest had or could
# give under the previous paragraph, plus a right to possession of the
# Corresponding Source of the work from the predecessor in interest, if
# the predecessor has it or can get it with reasonable efforts.
#
#   You may not impose any further restrictions on the exercise of the
# rights granted or affirmed under this License.  For example, you may
# not impose a license fee, royalty, or other charge for exercise of
# rights granted under this License, and you may not initiate litigation
# (including a cross-claim or counterclaim in a lawsuit) alleging that
# any patent claim is infringed by making, using, selling, offering for
# sale, or importing the Program or any portion of it.
#
#   11. Patents.
#
#   A "contributor" is a copyright holder who authorizes use under this
# License of the Program or a work on which the Program is based.  The
# work thus licensed is called the contributor's "contributor version".
#
#   A contributor's "essential patent claims" are all patent claims
# owned or controlled by the contributor, whether already acquired or
# hereafter acquired, that would be infringed by some manner, permitted
# by this License, of making, using, or selling its contributor version,
# but do not include claims that would be infringed only as a
# consequence of further modification of the contributor version.  For
# purposes of this definition, "control" includes the right to grant
# patent sublicenses in a manner consistent with the requirements of
# this License.
#
#   Each contributor grants you a non-exclusive, worldwide, royalty-free
# patent license under the contributor's essential patent claims, to
# make, use, sell, offer for sale, import and otherwise run, modify and
# propagate the contents of its contributor version.
#
#   In the following three paragraphs, a "patent license" is any express
# agreement or commitment, however denominated, not to enforce a patent
# (such as an express permission to practice a patent or covenant not to
# sue for patent infringement).  To "grant" such a patent license to a
# party means to make such an agreement or commitment not to enforce a
# patent against the party.
#
#   If you convey a covered work, knowingly relying on a patent license,
# and the Corresponding Source of the work is not available for anyone
# to copy, free of charge and under the terms of this License, through a
# publicly available network server or other readily accessible means,
# then you must either (1) cause the Corresponding Source to be so
# available, or (2) arrange to deprive yourself of the benefit of the
# patent license for this particular work, or (3) arrange, in a manner
# consistent with the requirements of this License, to extend the patent
# license to downstream recipients.  "Knowingly relying" means you have
# actual knowledge that, but for the patent license, your conveying the
# covered work in a country, or your recipient's use of the covered work
# in a country, would infringe one or more identifiable patents in that
# country that you have reason to believe are valid.
#
#   If, pursuant to or in connection with a single transaction or
# arrangement, you convey, or propagate by procuring conveyance of, a
# covered work, and grant a patent license to some of the parties
# receiving the covered work authorizing them to use, propagate, modify
# or convey a specific copy of the covered work, then the patent license
# you grant is automatically extended to all recipients of the covered
# work and works based on it.
#
#   A patent license is "discriminatory" if it does not include within
# the scope of its coverage, prohibits the exercise of, or is
# conditioned on the non-exercise of one or more of the rights that are
# specifically granted under this License.  You may not convey a covered
# work if you are a party to an arrangement with a third party that is
# in the business of distributing software, under which you make payment
# to the third party based on the extent of your activity of conveying
# the work, and under which the third party grants, to any of the
# parties who would receive the covered work from you, a discriminatory
# patent license (a) in connection with copies of the covered work
# conveyed by you (or copies made from those copies), or (b) primarily
# for and in connection with specific products or compilations that
# contain the covered work, unless you entered into that arrangement,
# or that patent license was granted, prior to 28 March 2007.
#
#   Nothing in this License shall be construed as excluding or limiting
# any implied license or other defenses to infringement that may
# otherwise be available to you under applicable patent law.
#
#   12. No Surrender of Others' Freedom.
#
#   If conditions are imposed on you (whether by court order, agreement or
# otherwise) that contradict the conditions of this License, they do not
# excuse you from the conditions of this License.  If you cannot convey a
# covered work so as to satisfy simultaneously your obligations under this
# License and any other pertinent obligations, then as a consequence you may
# not convey it at all.  For example, if you agree to terms that obligate you
# to collect a royalty for further conveying from those to whom you convey
# the Program, the only way you could satisfy both those terms and this
# License would be to refrain entirely from conveying the Program.
#
#   13. Use with the GNU Affero General Public License.
#
#   Notwithstanding any other provision of this License, you have
# permission to link or combine any covered work with a work licensed
# under version 3 of the GNU Affero General Public License into a single
# combined work, and to convey the resulting work.  The terms of this
# License will continue to apply to the part which is the covered work,
# but the special requirements of the GNU Affero General Public License,
# section 13, concerning interaction through a network will apply to the
# combination as such.
#
#   14. Revised Versions of this License.
#
#   The Free Software Foundation may publish revised and/or new versions of
# the GNU General Public License from time to time.  Such new versions will
# be similar in spirit to the present version, but may differ in detail to
# address new problems or concerns.
#
#   Each version is given a distinguishing version number.  If the
# Program specifies that a certain numbered version of the GNU General
# Public License "or any later version" applies to it, you have the
# option of following the terms and conditions either of that numbered
# version or of any later version published by the Free Software
# Foundation.  If the Program does not specify a version number of the
# GNU General Public License, you may choose any version ever published
# by the Free Software Foundation.
#
#   If the Program specifies that a proxy can decide which future
# versions of the GNU General Public License can be used, that proxy's
# public statement of acceptance of a version permanently authorizes you
# to choose that version for the Program.
#
#   Later license versions may give you additional or different
# permissions.  However, no additional obligations are imposed on any
# author or copyright holder as a result of your choosing to follow a
# later version.
#
#   15. Disclaimer of Warranty.
#
#   THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
# APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
# HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
# OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
# IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
# ALL NECESSARY SERVICING, REPAIR OR CORRECTION.
#
#   16. Limitation of Liability.
#
#   IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
# WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
# THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
# GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
# USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
# DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
# PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
# EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
#   17. Interpretation of Sections 15 and 16.
#
#   If the disclaimer of warranty and limitation of liability provided
# above cannot be given local legal effect according to their terms,
# reviewing courts shall apply local law that most closely approximates
# an absolute waiver of all civil liability in connection with the
# Program, unless a warranty or assumption of liability accompanies a
# copy of the Program in return for a fee.
#
#                      END OF TERMS AND CONDITIONS
#
#             How to Apply These Terms to Your New Programs
#
#   If you develop a new program, and you want it to be of the greatest
# possible use to the public, the best way to achieve this is to make it
# free software which everyone can redistribute and change under these terms.
#
#   To do so, attach the following notices to the program.  It is safest
# to attach them to the start of each source file to most effectively
# state the exclusion of warranty; and each file should have at least
# the "copyright" line and a pointer to where the full notice is found.
#
#     {one line to give the program's name and a brief idea of what it does.}
#     Copyright (C) {year}  {name of author}
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Also add information on how to contact you by electronic and paper mail.
#
#   If the program does terminal interaction, make it output a short
# notice like this when it starts in an interactive mode:
#
#     {project}  Copyright (C) {year}  {fullname}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.
#
# The hypothetical commands `show w' and `show c' should show the appropriate
# parts of the General Public License.  Of course, your program's commands
# might be different; for a GUI interface, you would use an "about box".
#
#   You should also get your employer (if you work as a programmer) or school,
# if any, to sign a "copyright disclaimer" for the program, if necessary.
# For more information on this, and how to apply and follow the GNU GPL, see
# <http://www.gnu.org/licenses/>.
#
#   The GNU General Public License does not permit incorporating your program
# into proprietary programs.  If your program is a subroutine library, you
# may consider it more useful to permit linking proprietary applications with
# the library.  If this is what you want to do, use the GNU Lesser General
# Public License instead of this License.  But first, please read
# <http://www.gnu.org/philosophy/why-not-lgpl.html>.

# ------------------------------------------------------------------------------
#  Usage:
#  python normative.py -m [maskfile] -k [number of CV folds] -c <covariates>
#                      -t [test covariates] -r [test responses] <infile>
#
#  Either the -k switch or -t switch should be specified, but not both.
#  If -t is selected, a set of responses should be provided with the -r switch
#
#  Written by A. Marquand
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import argparse
import pickle
import glob

from sklearn.model_selection import KFold
from pathlib import Path

try:  # run as a package if installed
    from pcntoolkit import configs
    from pcntoolkit.dataio import fileio
    from pcntoolkit.normative_model.norm_utils import norm_init
    from pcntoolkit.util.utils import compute_pearsonr, CustomCV, explained_var
    from pcntoolkit.util.utils import compute_MSLL, scaler, get_package_versions
except ImportError:
    pass

    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)
        # sys.path.append(os.path.join(path,'normative_model'))
    del path

    import configs
    # from dataio import fileio
    #
    # from util.utils import compute_pearsonr, CustomCV, explained_var, compute_MSLL
    # from util.utils import scaler, get_package_versions
    # from normative_model.norm_utils import norm_init

PICKLE_PROTOCOL = configs.PICKLE_PROTOCOL


def load_response_vars(datafile, maskfile=None, vol=True):
    """
    Load response variables from file. This will load the data and mask it if
    necessary. If the data is in ascii format it will be converted into a numpy
    array. If the data is in neuroimaging format it will be reshaped into a
    2D array (subjects x variables) and a mask will be created if necessary.

    :param datafile: File containing the response variables
    :param maskfile: Mask file (nifti only)
    :param vol: If True, load the data as a 4D volume (nifti only)
    :returns Y: Response variables
    :returns volmask: Mask file (nifti only)
    """

    if fileio.file_type(datafile) == 'nifti':
        dat = fileio.load_nifti(datafile, vol=vol)
        volmask = fileio.create_mask(dat, mask=maskfile)
        Y = fileio.vol2vec(dat, volmask).T
    else:
        Y = fileio.load(datafile)
        volmask = None
        if fileio.file_type(datafile) == 'cifti':
            Y = Y.T

    return Y, volmask


def get_args(*args):
    """
    Parse command line arguments for normative modeling

    :param args: command line arguments
    :returns respfile: response variables for the normative model
    :returns maskfile: mask used to apply to the data (nifti only)
    :returns covfile: covariates used to predict the response variable
    :returns cvfolds: Number of cross-validation folds
    :returns testcov: Test covariates
    :returns testresp: Test responses
    :returns func: Function to call
    :returns alg: Algorithm for normative model
    :returns configparam: Parameters controlling the estimation algorithm
    :returns kw_args: Additional keyword arguments
    """

    # parse arguments
    parser = argparse.ArgumentParser(description="Normative Modeling")
    parser.add_argument("responses")
    parser.add_argument("-f", help="Function to call", dest="func",
                        default="estimate")
    parser.add_argument("-m", help="mask file", dest="maskfile", default=None)
    parser.add_argument("-c", help="covariates file", dest="covfile",
                        default=None)
    parser.add_argument("-k", help="cross-validation folds", dest="cvfolds",
                        default=None)
    parser.add_argument("-t", help="covariates (test data)", dest="testcov",
                        default=None)
    parser.add_argument("-r", help="responses (test data)", dest="testresp",
                        default=None)
    parser.add_argument("-a", help="algorithm", dest="alg", default="gpr")
    parser.add_argument("-x", help="algorithm specific config options",
                        dest="configparam", default=None)
    # parser.add_argument('-s', action='store_false',
    #                 help="Flag to skip standardization.", dest="standardize")
    parser.add_argument("keyword_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Process required  arguemnts
    wdir = os.path.realpath(os.path.curdir)
    respfile = os.path.join(wdir, args.responses)
    if args.covfile is None:
        raise ValueError("No covariates specified")
    else:
        covfile = args.covfile

    # Process optional arguments
    if args.maskfile is None:
        maskfile = None
    else:
        maskfile = os.path.join(wdir, args.maskfile)
    if args.testcov is None and args.cvfolds is not None:
        testcov = None
        testresp = None
        cvfolds = int(args.cvfolds)
        print("Running under " + str(cvfolds) + " fold cross-validation.")
    else:
        print("Test covariates specified")
        testcov = args.testcov
        cvfolds = None
        if args.testresp is None:
            testresp = None
            print("No test response variables specified")
        else:
            testresp = args.testresp
        if args.cvfolds is not None:
            print("Ignoring cross-valdation specification (test data given)")

    # Process addtional keyword arguments. These are always added as strings
    kw_args = {}
    for kw in args.keyword_args:
        kw_arg = kw.split('=')

        exec("kw_args.update({'" + kw_arg[0] + "' : " +
             "'" + str(kw_arg[1]) + "'" + "})")

    return respfile, maskfile, covfile, cvfolds, \
        testcov, testresp, args.func, args.alg, \
        args.configparam, kw_args


def evaluate(Y, Yhat, S2=None, mY=None, sY=None, nlZ=None, nm=None, Xz_tr=None, alg=None,
             metrics=['Rho', 'RMSE', 'SMSE', 'EXPV', 'MSLL']):
    ''' Compute error metrics
    This function will compute error metrics based on a set of predictions Yhat
    and a set of true response variables Y, namely:

    * Rho: Pearson correlation
    * RMSE: root mean squared error
    * SMSE: standardized mean squared error
    * EXPV: explained variance

    If the predictive variance is also specified the log loss will be computed
    (which also takes into account the predictive variance). If the mean and
    standard deviation are also specified these will be used to standardize
    this, yielding the mean standardized log loss

    :param Y: N x P array of true response variables
    :param Yhat: N x P array of predicted response variables
    :param S2: predictive variance
    :param mY: mean of the training set
    :param sY: standard deviation of the training set

    :returns metrics: evaluation metrics

    '''

    feature_num = Y.shape[1]

    # Remove metrics that cannot be computed with only a single data point
    if Y.shape[0] == 1:
        if 'MSLL' in metrics:
            metrics.remove('MSLL')
        if 'SMSE' in metrics:
            metrics.remove('SMSE')

    # find and remove bad variables from the response variables
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]

    MSE = np.mean((Y - Yhat)**2, axis=0)

    results = dict()

    if 'RMSE' in metrics:
        RMSE = np.sqrt(MSE)
        results['RMSE'] = RMSE

    if 'Rho' in metrics:
        Rho = np.zeros(feature_num)
        pRho = np.ones(feature_num)
        Rho[nz], pRho[nz] = compute_pearsonr(Y[:, nz], Yhat[:, nz])
        results['Rho'] = Rho
        results['pRho'] = pRho

    if 'SMSE' in metrics:
        SMSE = np.zeros_like(MSE)
        SMSE[nz] = MSE[nz] / np.var(Y[:, nz], axis=0)
        results['SMSE'] = SMSE

    if 'EXPV' in metrics:
        EXPV = np.zeros(feature_num)
        EXPV[nz] = explained_var(Y[:, nz], Yhat[:, nz])
        results['EXPV'] = EXPV

    if 'MSLL' in metrics:
        if ((S2 is not None) and (mY is not None) and (sY is not None)):
            MSLL = np.zeros(feature_num)
            MSLL[nz] = compute_MSLL(Y[:, nz], Yhat[:, nz], S2[:, nz],
                                    mY.reshape(-1, 1).T,
                                    (sY**2).reshape(-1, 1).T)
            results['MSLL'] = MSLL

    if 'NLL' in metrics:
        results['NLL'] = nlZ

    if 'BIC' in metrics:
        if hasattr(getattr(nm, alg), 'hyp'):
            n = Xz_tr.shape[0]
            k = len(getattr(nm, alg).hyp)
            BIC = k * np.log(n) + 2 * nlZ
            results['BIC'] = BIC

    return results


def save_results(respfile, Yhat, S2, maskvol, Z=None, Y=None, outputsuffix=None,
                 results=None, save_path=''):
    """
    Writes the results of the normative model to disk.

    Parameters:
    respfile (str): The response variables file.
    Yhat (np.array): The predicted response variables.
    S2 (np.array): The predictive variance.
    maskvol (np.array): The mask volume.
    Z (np.array, optional): The latent variable. Defaults to None.
    Y (np.array, optional): The observed response variables. Defaults to None.
    outputsuffix (str, optional): The suffix to append to the output files. Defaults to None.
    results (dict, optional): The results of the normative model. Defaults to None.
    save_path (str, optional): The directory to save the results to. Defaults to ''.

    Returns:
    None
    """

    print("Writing outputs ...")
    if respfile is None:
        exfile = None
        file_ext = '.pkl'
    else:
        if fileio.file_type(respfile) == 'cifti' or \
           fileio.file_type(respfile) == 'nifti':
            exfile = respfile
        else:
            exfile = None
        file_ext = fileio.file_extension(respfile)

    if outputsuffix is not None:
        ext = str(outputsuffix) + file_ext
    else:
        ext = file_ext

    fileio.save(Yhat, os.path.join(save_path, 'yhat' + ext), example=exfile,
                mask=maskvol)
    fileio.save(S2, os.path.join(save_path, 'ys2' + ext), example=exfile,
                mask=maskvol)
    if Z is not None:
        fileio.save(Z, os.path.join(save_path, 'Z' + ext), example=exfile,
                    mask=maskvol)
    if Y is not None:
        fileio.save(Y, os.path.join(save_path, 'Y' + ext), example=exfile,
                    mask=maskvol)
    if results is not None:
        for metric in list(results.keys()):
            if (metric == 'NLL' or metric == 'BIC') and file_ext == '.nii.gz':
                fileio.save(results[metric], os.path.join(save_path, metric + str(outputsuffix) + '.pkl'),
                            example=exfile, mask=maskvol)
            else:
                fileio.save(results[metric], os.path.join(save_path, metric + ext),
                            example=exfile, mask=maskvol)


def estimate(covfile, respfile, **kwargs):
    """ Estimate a normative model

    This will estimate a model in one of two settings according to
    theparticular parameters specified (see below)

    * under k-fold cross-validation.
      requires respfile, covfile and cvfolds>=2
    * estimating a training dataset then applying to a second test dataset.
      requires respfile, covfile, testcov and testresp.
    * estimating on a training dataset ouput of forward maps mean and se.
      requires respfile, covfile and testcov

    The models are estimated on the basis of data stored on disk in ascii or
    neuroimaging data formats (nifti or cifti). Ascii data should be in
    tab or space delimited format with the number of subjects in rows and the
    number of variables in columns. Neuroimaging data will be reshaped
    into the appropriate format

    Basic usage::

        estimate(covfile, respfile, [extra_arguments])

    where the variables are defined below. Note that either the cfolds
    parameter or (testcov, testresp) should be specified, but not both.

    :param respfile: response variables for the normative model
    :param covfile: covariates used to predict the response variable
    :param maskfile: mask used to apply to the data (nifti only)
    :param cvfolds: Number of cross-validation folds
    :param testcov: Test covariates
    :param testresp: Test responses
    :param alg: Algorithm for normative model
    :param configparam: Parameters controlling the estimation algorithm
    :param saveoutput: Save the output to disk? Otherwise returned as arrays
    :param outputsuffix: Text string to add to the output filenames
    :param inscaler: Scaling approach for input covariates, could be 'None' (Default),
                    'standardize', 'minmax', or 'robminmax'.
    :param outscaler: Scaling approach for output responses, could be 'None' (Default),
                    'standardize', 'minmax', or 'robminmax'.

    All outputs are written to disk in the same format as the input. These are:

    :outputs: * yhat - predictive mean
              * ys2 - predictive variance
              * nm - normative model
              * Z - deviance scores
              * Rho - Pearson correlation between true and predicted responses
              * pRho - parametric p-value for this correlation
              * rmse - root mean squared error between true/predicted responses
              * smse - standardised mean squared error

    The outputsuffix may be useful to estimate multiple normative models in the
    same directory (e.g. for custom cross-validation schemes)
    """

    # parse keyword arguments
    maskfile = kwargs.pop('maskfile', None)
    cvfolds = kwargs.pop('cvfolds', None)
    testcov = kwargs.pop('testcov', None)
    testresp = kwargs.pop('testresp', None)
    alg = kwargs.pop('alg', 'gpr')
    outputsuffix = kwargs.pop('outputsuffix', 'estimate')
    # Making sure there is only one
    outputsuffix = "_" + outputsuffix.replace("_", "")
    # '_' is in the outputsuffix to
    # avoid file name parsing problem.
    inscaler = kwargs.pop('inscaler', 'None')
    outscaler = kwargs.pop('outscaler', 'None')
    warp = kwargs.get('warp', None)

    # convert from strings if necessary
    saveoutput = kwargs.pop('saveoutput', 'True')
    if type(saveoutput) is str:
        saveoutput = saveoutput == 'True'
    savemodel = kwargs.pop('savemodel', 'False')
    if type(savemodel) is str:
        savemodel = savemodel == 'True'

    if savemodel and not os.path.isdir('Models'):
        os.mkdir('Models')

    # which output metrics to compute
    metrics = ['Rho', 'RMSE', 'SMSE', 'EXPV', 'MSLL', 'NLL', 'BIC']

    # load data
    print("Processing data in " + respfile)
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    Nmod = Y.shape[1]

    if (testcov is not None) and (cvfolds is None):  # a separate test dataset

        run_cv = False
        cvfolds = 1
        Xte = fileio.load(testcov)
        if len(Xte.shape) == 1:
            Xte = Xte[:, np.newaxis]
        if testresp is not None:
            Yte, testmask = load_response_vars(testresp, maskfile)
            if len(Yte.shape) == 1:
                Yte = Yte[:, np.newaxis]
        else:
            sub_te = Xte.shape[0]
            Yte = np.zeros([sub_te, Nmod])

        # treat as a single train-test split
        testids = range(X.shape[0], X.shape[0]+Xte.shape[0])
        splits = CustomCV((range(0, X.shape[0]),), (testids,))

        Y = np.concatenate((Y, Yte), axis=0)
        X = np.concatenate((X, Xte), axis=0)

    else:
        run_cv = True
        # we are running under cross-validation
        splits = KFold(n_splits=cvfolds, shuffle=True)
        testids = range(0, X.shape[0])
        if alg == 'hbr':
            trbefile = kwargs.get('trbefile', None)
            if trbefile is not None:
                be = fileio.load(trbefile)
                if len(be.shape) == 1:
                    be = be[:, np.newaxis]
            else:
                print('No batch-effects file! Initilizing all as zeros!')
                be = np.zeros([X.shape[0], 1])

    # find and remove bad variables from the response variables
    # note: the covariates are assumed to have already been checked
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]

    # run cross-validation loop
    Yhat = np.zeros_like(Y)
    S2 = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    nlZ = np.zeros((Nmod, cvfolds))

    scaler_resp = []
    scaler_cov = []
    mean_resp = []  # this is just for computing MSLL
    std_resp = []  # this is just for computing MSLL

    if warp is not None:
        Ywarp = np.zeros_like(Yhat)

        # for warping we need to compute metrics separately for each fold
        results_folds = dict()
        for m in metrics:
            results_folds[m] = np.zeros((Nmod, cvfolds))

    for idx in enumerate(splits.split(X)):

        fold = idx[0]
        tr = idx[1][0]
        ts = idx[1][1]

        # standardize responses and covariates, ignoring invalid entries
        iy_tr, jy_tr = np.ix_(tr, nz)
        iy_ts, jy_ts = np.ix_(ts, nz)
        mY = np.mean(Y[iy_tr, jy_tr], axis=0)
        sY = np.std(Y[iy_tr, jy_tr], axis=0)
        mean_resp.append(mY)
        std_resp.append(sY)

        if inscaler in ['standardize', 'minmax', 'robminmax']:
            X_scaler = scaler(inscaler)
            Xz_tr = X_scaler.fit_transform(X[tr, :])
            Xz_ts = X_scaler.transform(X[ts, :])
            scaler_cov.append(X_scaler)
        else:
            Xz_tr = X[tr, :]
            Xz_ts = X[ts, :]

        if outscaler in ['standardize', 'minmax', 'robminmax']:
            Y_scaler = scaler(outscaler)
            Yz_tr = Y_scaler.fit_transform(Y[iy_tr, jy_tr])
            scaler_resp.append(Y_scaler)
        else:
            Yz_tr = Y[iy_tr, jy_tr]

        if (run_cv == True and alg == 'hbr'):
            fileio.save(be[tr, :], 'be_kfold_tr_tempfile.pkl')
            fileio.save(be[ts, :], 'be_kfold_ts_tempfile.pkl')
            kwargs['trbefile'] = 'be_kfold_tr_tempfile.pkl'
            kwargs['tsbefile'] = 'be_kfold_ts_tempfile.pkl'

        # estimate the models for all response variables
        for i in range(0, len(nz)):
            print("Estimating model ", i+1, "of", len(nz))
            nm = norm_init(Xz_tr, Yz_tr[:, i], alg=alg, **kwargs)

            try:
                nm = nm.estimate(Xz_tr, Yz_tr[:, i], **kwargs)
                yhat, s2 = nm.predict(Xz_ts, Xz_tr, Yz_tr[:, i], **kwargs)

                if savemodel:
                    nm.save('Models/NM_' + str(fold) + '_' + str(nz[i]) +
                            outputsuffix + '.pkl')

                if outscaler == 'standardize':
                    Yhat[ts, nz[i]] = Y_scaler.inverse_transform(yhat, index=i)
                    S2[ts, nz[i]] = s2 * sY[i]**2
                elif outscaler in ['minmax', 'robminmax']:
                    Yhat[ts, nz[i]] = Y_scaler.inverse_transform(yhat, index=i)
                    S2[ts, nz[i]] = s2 * (Y_scaler.max[i] - Y_scaler.min[i])**2
                else:
                    Yhat[ts, nz[i]] = yhat
                    S2[ts, nz[i]] = s2

                nlZ[nz[i], fold] = nm.neg_log_lik

                if (run_cv or testresp is not None):
                    if warp is not None:
                        # TODO: Warping for scaled data
                        if outscaler is not None and outscaler != 'None':
                            raise ValueError("outscaler not yet supported warping")
                        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1]
                        Ywarp[ts, nz[i]] = nm.blr.warp.f(
                            Y[ts, nz[i]], warp_param)
                        Ytest = Ywarp[ts, nz[i]]

                        # Save warped mean of the training data (for MSLL)
                        yw = nm.blr.warp.f(Y[tr, nz[i]], warp_param)

                        # create arrays for evaluation
                        Yhati = Yhat[ts, nz[i]]
                        Yhati = Yhati[:, np.newaxis]
                        S2i = S2[ts, nz[i]]
                        S2i = S2i[:, np.newaxis]

                        # evaluate and save results
                        mf = evaluate(Ytest[:, np.newaxis], Yhati, S2=S2i,
                                      mY=np.mean(yw), sY=np.std(yw),
                                      nlZ=nm.neg_log_lik, nm=nm, Xz_tr=Xz_tr,
                                      alg=alg, metrics=metrics)
                        for k in metrics:
                            results_folds[k][nz[i]][fold] = mf[k]
                    else:
                        Ytest = Y[ts, nz[i]]

                    if alg == 'hbr':
                        if outscaler in ['standardize', 'minmax', 'robminmax']:
                            Ytestz = Y_scaler.transform(
                                Ytest.reshape(-1, 1), index=i)
                        else:
                            Ytestz = Ytest.reshape(-1, 1)
                        Z[ts, nz[i]] = nm.get_mcmc_zscores(
                            Xz_ts, Ytestz, **kwargs)
                    else:
                        Z[ts, nz[i]] = (Ytest - Yhat[ts, nz[i]]) / \
                            np.sqrt(S2[ts, nz[i]])

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("Model ", i+1, "of", len(nz),
                      "FAILED!..skipping and writing NaN to outputs")
                print("Exception:")
                print(e)
                print(exc_type, fname, exc_tb.tb_lineno)

                Yhat[ts, nz[i]] = float('nan')
                S2[ts, nz[i]] = float('nan')
                nlZ[nz[i], fold] = float('nan')
                if testcov is None:
                    Z[ts, nz[i]] = float('nan')
                else:
                    if testresp is not None:
                        Z[ts, nz[i]] = float('nan')

    if savemodel:
        print('Saving model meta-data...')
        v = get_package_versions()
        with open('Models/meta_data.md', 'wb') as file:
            pickle.dump({'valid_voxels': nz, 'fold_num': cvfolds,
                         'mean_resp': mean_resp, 'std_resp': std_resp,
                         'scaler_cov': scaler_cov, 'scaler_resp': scaler_resp,
                         'regressor': alg, 'inscaler': inscaler,
                         'outscaler': outscaler, 'versions': v},
                        file, protocol=PICKLE_PROTOCOL)

    # compute performance metrics
    if (run_cv or testresp is not None):
        print("Evaluating the model ...")
        if warp is None:
            results = evaluate(Y[testids, :], Yhat[testids, :],
                               S2=S2[testids, :], mY=mean_resp[0],
                               sY=std_resp[0], nlZ=nlZ, nm=nm, Xz_tr=Xz_tr, alg=alg,
                               metrics=metrics)
        else:
            # for warped data we just aggregate across folds
            results = dict()
            for m in ['Rho', 'RMSE', 'SMSE', 'EXPV', 'MSLL']:
                results[m] = np.mean(results_folds[m], axis=1)
            results['NLL'] = results_folds['NLL']
            results['BIC'] = results_folds['BIC']

    # Set writing options
    if saveoutput:
        if (run_cv or testresp is not None):
            save_results(respfile, Yhat[testids, :], S2[testids, :], maskvol,
                         Z=Z[testids, :], results=results,
                         outputsuffix=outputsuffix)

        else:
            save_results(respfile, Yhat[testids, :], S2[testids, :], maskvol,
                         outputsuffix=outputsuffix)

    else:
        if (run_cv or testresp is not None):
            output = (Yhat[testids, :], S2[testids, :], nm, Z[testids, :],
                      results)
        else:
            output = (Yhat[testids, :], S2[testids, :], nm)

        return output


def fit(covfile, respfile, **kwargs):
    """
    Fits a normative model to the data.

    Parameters:
    covfile (str): The path to the covariates file.
    respfile (str): The path to the response variables file.
    maskfile (str, optional): The path to the mask file. Defaults to None.
    alg (str, optional): The algorithm to use. Defaults to 'gpr'.
    savemodel (bool, optional): Whether to save the model. Defaults to True.
    outputsuffix (str, optional): The suffix to append to the output files. Defaults to 'fit'.
    inscaler (str, optional): The scaler to use for the input data. Defaults to 'None'.
    outscaler (str, optional): The scaler to use for the output data. Defaults to 'None'.

    Returns:
    None
    """

    # parse keyword arguments
    maskfile = kwargs.pop('maskfile', None)
    alg = kwargs.pop('alg', 'gpr')
    savemodel = kwargs.pop('savemodel', 'True') == 'True'
    outputsuffix = kwargs.pop('outputsuffix', 'fit')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inscaler = kwargs.pop('inscaler', 'None')
    outscaler = kwargs.pop('outscaler', 'None')

    if savemodel and not os.path.isdir('Models'):
        os.mkdir('Models')

    # load data
    print("Processing data in " + respfile)
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    # find and remove bad variables from the response variables
    # note: the covariates are assumed to have already been checked
    nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                 np.var(Y, axis=0) != 0))[0]

    scaler_resp = []
    scaler_cov = []
    mean_resp = []  # this is just for computing MSLL
    std_resp = []   # this is just for computing MSLL

    # standardize responses and covariates, ignoring invalid entries
    mY = np.mean(Y[:, nz], axis=0)
    sY = np.std(Y[:, nz], axis=0)
    mean_resp.append(mY)
    std_resp.append(sY)

    if inscaler in ['standardize', 'minmax', 'robminmax']:
        X_scaler = scaler(inscaler)
        Xz = X_scaler.fit_transform(X)
        scaler_cov.append(X_scaler)
    else:
        Xz = X

    if outscaler in ['standardize', 'minmax', 'robminmax']:
        Yz = np.zeros_like(Y)
        Y_scaler = scaler(outscaler)
        Yz[:, nz] = Y_scaler.fit_transform(Y[:, nz])
        scaler_resp.append(Y_scaler)
    else:
        Yz = Y

    # estimate the models for all subjects
    for i in range(0, len(nz)):
        print("Estimating model ", i+1, "of", len(nz))
        nm = norm_init(Xz, Yz[:, nz[i]], alg=alg, **kwargs)
        nm = nm.estimate(Xz, Yz[:, nz[i]], **kwargs)

        if savemodel:
            nm.save('Models/NM_' + str(0) + '_' + str(nz[i]) + outputsuffix +
                    '.pkl')

    if savemodel:
        print('Saving model meta-data...')
        v = get_package_versions()
        with open('Models/meta_data.md', 'wb') as file:
            pickle.dump({'valid_voxels': nz,
                         'mean_resp': mean_resp, 'std_resp': std_resp,
                         'scaler_cov': scaler_cov, 'scaler_resp': scaler_resp,
                         'regressor': alg, 'inscaler': inscaler,
                         'outscaler': outscaler, 'versions': v},
                        file, protocol=PICKLE_PROTOCOL)

    return nm


def predict(covfile, respfile, maskfile=None, **kwargs):
    '''
    Make predictions on the basis of a pre-estimated normative model
    If only the covariates are specified then only predicted mean and variance
    will be returned. If the test responses are also specified then quantities
    That depend on those will also be returned (Z scores and error metrics)

    Basic usage::

        predict(covfile, [extra_arguments])

    where the variables are defined below.

    :param covfile: test covariates used to predict the response variable
    :param respfile: test response variables for the normative model
    :param maskfile: mask used to apply to the data (nifti only)
    :param model_path: Directory containing the normative model and metadata.
     When using parallel prediction, do not pass the model path. It will be
     automatically decided.
    :param outputsuffix: Text string to add to the output filenames
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param fold: which cross-validation fold to use (default = 0)
    :param fold: list of model IDs to predict (if not specified all are computed)
    :param return_y: return the (transformed) response variable (default = False)

    All outputs are written to disk in the same format as the input. These are:

    :outputs: * Yhat - predictive mean
              * S2 - predictive variance
              * Z - Z scores
              * Y - response variable (if return_y is True)
    '''

    model_path = kwargs.pop('model_path', 'Models')
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    outputsuffix = kwargs.pop('outputsuffix', 'predict')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    alg = kwargs.pop('alg')
    fold = kwargs.pop('fold', 0)
    models = kwargs.pop('models', None)
    return_y = kwargs.pop('return_y', False)

    if alg == 'gpr':
        raise ValueError("gpr is not supported with predict()")

    if respfile is not None and not os.path.exists(respfile):
        print("Response file does not exist. Only returning predictions")
        respfile = None
    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            inscaler = meta_data['inscaler']
            outscaler = meta_data['outscaler']
            mY = meta_data['mean_resp']
            sY = meta_data['std_resp']
            scaler_cov = meta_data['scaler_cov']
            scaler_resp = meta_data['scaler_resp']
            meta_data = True
        else:
            print("No meta-data file is found!")
            inscaler = 'None'
            outscaler = 'None'
            meta_data = False

    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1

    # load data
    print("Loading data ...")
    X = fileio.load(covfile)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    sample_num = X.shape[0]
    if models is not None:
        feature_num = len(models)
    else:
        feature_num = len(glob.glob(os.path.join(model_path, 'NM_' + str(fold) +
                                                 '_*' + inputsuffix + '.pkl')))
        models = range(feature_num)

    Yhat = np.zeros([sample_num, feature_num])
    S2 = np.zeros([sample_num, feature_num])
    Z = np.zeros([sample_num, feature_num])

    if inscaler in ['standardize', 'minmax', 'robminmax']:
        Xz = scaler_cov[fold].transform(X)
    else:
        Xz = X

    # estimate the models for all variabels
    # TODO Z-scores adaptation for SHASH HBR
    for i, m in enumerate(models):
        print("Prediction by model ", i+1, "of", feature_num)
        nm = norm_init(Xz)
        nm = nm.load(os.path.join(model_path, 'NM_' + str(fold) + '_' +
                                  str(m) + inputsuffix + '.pkl'))
        if (alg != 'hbr' or nm.configs['transferred'] == False):
            yhat, s2 = nm.predict(Xz, **kwargs)
        else:
            tsbefile = kwargs.get('tsbefile')
            batch_effects_test = fileio.load(tsbefile)
            yhat, s2 = nm.predict_on_new_sites(Xz, batch_effects_test)

        if outscaler == 'standardize':
            Yhat[:, i] = scaler_resp[fold].inverse_transform(yhat, index=i)
            S2[:, i] = s2.squeeze() * sY[fold][i]**2
        elif outscaler in ['minmax', 'robminmax']:
            Yhat[:, i] = scaler_resp[fold].inverse_transform(yhat, index=i)
            S2[:, i] = s2 * (scaler_resp[fold].max[i] -
                             scaler_resp[fold].min[i])**2
        else:
            Yhat[:, i] = yhat.squeeze()
            S2[:, i] = s2.squeeze()

    if respfile is None:
        save_results(None, Yhat, S2, None, outputsuffix=outputsuffix)

        return (Yhat, S2)

    else:
        Y, maskvol = load_response_vars(respfile, maskfile)
        if models is not None and len(Y.shape) > 1:
            Y = Y[:, models]
            if meta_data:
                # are we using cross-validation?
                if type(mY) is list:
                    mY = mY[fold][models]
                else:
                    mY = mY[models]
                if type(sY) is list:
                    sY = sY[fold][models]
                else:
                    sY = sY[models]

        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]

        # warp the targets?
        if alg == 'blr' and nm.blr.warp is not None:
            warp = True
            Yw = np.zeros_like(Y)
            for i, m in enumerate(models):
                nm = norm_init(Xz)
                nm = nm.load(os.path.join(model_path, 'NM_' + str(fold) + '_' +
                                          str(m) + inputsuffix + '.pkl'))

                warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1]
                Yw[:, i] = nm.blr.warp.f(Y[:, i], warp_param)
            Y = Yw
        else:
            warp = False

        Z = (Y - Yhat) / np.sqrt(S2)

        print("Evaluating the model ...")
        if meta_data and not warp:

            # results = evaluate(Y, Yhat, S2=S2, mY=mY, sY=sY)
        # else:
            results = evaluate(Y, Yhat, S2=S2,
                               metrics=['Rho', 'RMSE', 'SMSE', 'EXPV'])

        print("Evaluations Writing outputs ...")

        if return_y:
            save_results(respfile, Yhat, S2, maskvol, Z=Z, Y=Y,
                         outputsuffix=outputsuffix, results=results)
            return (Yhat, S2, Z, Y)
        else:
            save_results(respfile, Yhat, S2, maskvol, Z=Z,
                         outputsuffix=outputsuffix, results=results)
            return (Yhat, S2, Z)


def transfer(covfile, respfile, testcov=None, testresp=None, maskfile=None,
             **kwargs):
    '''
    Transfer learning on the basis of a pre-estimated normative model by using
    the posterior distribution over the parameters as an informed prior for
    new data. currently only supported for HBR.

    Basic usage::

        transfer(covfile, respfile [extra_arguments])

    where the variables are defined below.

    :param covfile: transfer covariates used to predict the response variable
    :param respfile: transfer response variables for the normative model
    :param maskfile: mask used to apply to the data (nifti only)
    :param testcov: Test covariates
    :param testresp: Test responses
    :param model_path: Directory containing the normative model and metadata
    :param trbefile: Training batch effects file
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id

    All outputs are written to disk in the same format as the input. These are:

    :outputs: * Yhat - predictive mean
              * S2 - predictive variance
              * Z - Z scores
    '''
    alg = kwargs.pop('alg').lower()

    if alg != 'hbr' and alg != 'blr':
        print('Model transfer function is only possible for HBR and BLR models.')
        return
    # testing should not be obligatory for HBR,
    # but should be for BLR (since it doesn't produce transfer models)
    elif (not 'model_path' in list(kwargs.keys())) or \
            (not 'trbefile' in list(kwargs.keys())):
        print(f'{kwargs=}')
        print('InputError: Some general mandatory arguments are missing.')
        return
    # hbr has one additional mandatory arguments
    elif alg == 'hbr':
        if (not 'output_path' in list(kwargs.keys())):
            print('InputError: Some mandatory arguments for hbr are missing.')
            return
        else:
            output_path = kwargs.pop('output_path', None)
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

    # for hbr, testing is not mandatory, for blr's predict/transfer it is. This will be an architectural choice.
    # or (testresp==None)
    elif alg == 'blr':
        if (testcov == None) or \
                (not 'tsbefile' in list(kwargs.keys())):
            print('InputError: Some mandatory arguments for blr are missing.')
            return
    # general arguments
    log_path = kwargs.pop('log_path', None)
    model_path = kwargs.pop('model_path')
    outputsuffix = kwargs.pop('outputsuffix', 'transfer')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    tsbefile = kwargs.pop('tsbefile', None)
    trbefile = kwargs.pop('trbefile', None)
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    fold = kwargs.pop('fold', 0)

    # for PCNonline automated parallel jobs loop
    count_jobsdone = kwargs.pop('count_jobsdone', 'False')
    if type(count_jobsdone) is str:
        count_jobsdone = count_jobsdone == 'True'

    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1

    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            inscaler = meta_data['inscaler']
            outscaler = meta_data['outscaler']
            scaler_cov = meta_data['scaler_cov']
            scaler_resp = meta_data['scaler_resp']
            meta_data = True
        else:
            print("No meta-data file is found!")
            inscaler = 'None'
            outscaler = 'None'
            meta_data = False

    # load adaptation data
    print("Loading data ...")
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    if inscaler in ['standardize', 'minmax', 'robminmax']:
        X = scaler_cov[0].transform(X)

    feature_num = Y.shape[1]
    mY = np.mean(Y, axis=0)
    sY = np.std(Y, axis=0)

    if outscaler in ['standardize', 'minmax', 'robminmax']:
        Y = scaler_resp[0].transform(Y)

    batch_effects_train = fileio.load(trbefile)

    # load test data
    if testcov is not None:
        # we have a separate test dataset
        Xte = fileio.load(testcov)
        if len(Xte.shape) == 1:
            Xte = Xte[:, np.newaxis]
        ts_sample_num = Xte.shape[0]
        if inscaler in ['standardize', 'minmax', 'robminmax']:
            Xte = scaler_cov[0].transform(Xte)

        if testresp is not None:
            Yte, testmask = load_response_vars(testresp, maskfile)
            if len(Yte.shape) == 1:
                Yte = Yte[:, np.newaxis]
        else:
            Yte = np.zeros([ts_sample_num, feature_num])

        if tsbefile is not None:
            batch_effects_test = fileio.load(tsbefile)
        else:
            batch_effects_test = np.zeros([Xte.shape[0], 2])
    else:
        ts_sample_num = 0

    Yhat = np.zeros([ts_sample_num, feature_num])
    S2 = np.zeros([ts_sample_num, feature_num])
    Z = np.zeros([ts_sample_num, feature_num])

    # estimate the models for all subjects
    for i in range(feature_num):

        if alg == 'hbr':
            print("Using HBR transform...")
            nm = norm_init(X)
            if batch_size is not None:  # when using normative_parallel
                print("Transferring model ", job_id*batch_size+i)
                nm = nm.load(os.path.join(model_path, 'NM_0_' +
                                          str(job_id*batch_size+i) + inputsuffix +
                                          '.pkl'))
            else:
                print("Transferring model ", i+1, "of", feature_num)
                nm = nm.load(os.path.join(model_path, 'NM_0_' + str(i) +
                                          inputsuffix + '.pkl'))

            nm = nm.estimate_on_new_sites(X, Y[:, i], batch_effects_train)
            if batch_size is not None:
                nm.save(os.path.join(output_path, 'NM_0_' +
                                     str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            else:
                nm.save(os.path.join(output_path, 'NM_0_' +
                                     str(i) + outputsuffix + '.pkl'))

            if testcov is not None:
                yhat, s2 = nm.predict_on_new_sites(Xte, batch_effects_test)

        # We basically use normative.predict script here.
        if alg == 'blr':
            print("Using BLR transform...")
            print("Transferring model ", i+1, "of", feature_num)
            nm = norm_init(X)
            nm = nm.load(os.path.join(model_path, 'NM_' + str(fold) + '_' +
                                      str(i) + inputsuffix + '.pkl'))

            # translate the syntax to what blr understands
            # first strip existing blr keyword arguments to avoid redundancy
            adapt_cov = kwargs.pop('adaptcovfile', None)
            adapt_res = kwargs.pop('adaptrespfile', None)
            adapt_vg = kwargs.pop('adaptvargroupfile', None)
            test_vg = kwargs.pop('testvargroupfile', None)
            if adapt_cov is not None or adapt_res is not None \
                    or adapt_vg is not None or test_vg is not None:
                print(
                    "Warning: redundant batch effect parameterisation. Using HBR syntax")

            yhat, s2 = nm.predict(Xte, X, Y[:, i],
                                  adaptcov=X,
                                  adaptresp=Y[:, i],
                                  adaptvargroup=batch_effects_train,
                                  testvargroup=batch_effects_test,
                                  **kwargs)

        if testcov is not None:
            if outscaler == 'standardize':
                Yhat[:, i] = scaler_resp[0].inverse_transform(
                    yhat.squeeze(), index=i)
                S2[:, i] = s2.squeeze() * sY[i]**2
            elif outscaler in ['minmax', 'robminmax']:
                Yhat[:, i] = scaler_resp[0].inverse_transform(yhat, index=i)
                S2[:, i] = s2 * (scaler_resp[0].max[i] -
                                 scaler_resp[0].min[i])**2
            else:
                Yhat[:, i] = yhat.squeeze()
                S2[:, i] = s2.squeeze()

    if testresp is None:
        save_results(respfile, Yhat, S2, maskvol, outputsuffix=outputsuffix)
        return (Yhat, S2)
    else:
        # warp the targets?
        if alg == 'blr' and nm.blr.warp is not None:
            warp = True
            Yw = np.zeros_like(Yte)
            for i in range(feature_num):
                nm = norm_init(Xte)
                nm = nm.load(os.path.join(model_path, 'NM_' + str(fold) + '_' +
                                          str(i) + inputsuffix + '.pkl'))

                warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1]
                Yw[:, i] = nm.blr.warp.f(Yte[:, i], warp_param)
            Yte = Yw
        else:
            warp = False

        # TODO Z-scores adaptation for SHASH HBR
        Z = (Yte - Yhat) / np.sqrt(S2)

        print("Evaluating the model ...")
        if meta_data and not warp:
            results = evaluate(Yte, Yhat, S2=S2, mY=mY, sY=sY)
        else:
            results = evaluate(Yte, Yhat, S2=S2,
                               metrics=['Rho', 'RMSE', 'SMSE', 'EXPV'])

        save_results(respfile, Yhat, S2, maskvol, Z=Z, results=results,
                     outputsuffix=outputsuffix)

        # Creates a file for every job succesfully completed (for tracking failed jobs).
        if count_jobsdone == True:
            done_path = os.path.join(log_path, str(job_id)+".jobsdone")
            Path(done_path).touch()

        return (Yhat, S2, Z)

    # Creates a file for every job succesfully completed (for tracking failed jobs).
    if count_jobsdone == True:
        done_path = os.path.join(log_path, str(job_id)+".jobsdone")
        Path(done_path).touch()


def extend(covfile, respfile, maskfile=None, **kwargs):
    '''
    This function extends an existing HBR model with data from new sites/scanners.

    Basic usage::

        extend(covfile, respfile [extra_arguments])

    where the variables are defined below.

    :param covfile: covariates for new data
    :param respfile: response variables for new data
    :param maskfile: mask used to apply to the data (nifti only)
    :param model_path: Directory containing the normative model and metadata
    :param trbefile: file address to batch effects file for new data
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param output_path: the path for saving the  the extended model
    :param informative_prior: use initial model prior or learn from scratch (default is False).
    :param generation_factor: see below

    generation factor refers to the number of samples generated for each
    combination of covariates and batch effects. Default is 10.


    All outputs are written to disk in the same format as the input.

    '''

    alg = kwargs.pop('alg')
    if alg != 'hbr':
        print('Model extention is only possible for HBR models.')
        return
    elif (not 'model_path' in list(kwargs.keys())) or \
        (not 'output_path' in list(kwargs.keys())) or \
            (not 'trbefile' in list(kwargs.keys())):
        print('InputError: Some mandatory arguments are missing.')
        return
    else:
        model_path = kwargs.pop('model_path')
        output_path = kwargs.pop('output_path')
        trbefile = kwargs.pop('trbefile')

    outputsuffix = kwargs.pop('outputsuffix', 'extend')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    informative_prior = kwargs.pop('informative_prior', 'False') == 'True'
    generation_factor = int(kwargs.pop('generation_factor', '10'))
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1

    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            if (meta_data['inscaler'] != 'None' or
                    meta_data['outscaler'] != 'None'):
                print('Models extention on scaled data is not possible!')
                return

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # load data
    print("Loading data ...")
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    batch_effects_train = fileio.load(trbefile)

    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    feature_num = Y.shape[1]

    # estimate the models for all subjects
    for i in range(feature_num):

        nm = norm_init(X)
        if batch_size is not None:  # when using nirmative_parallel
            print("Extending model ", job_id*batch_size+i)
            nm = nm.load(os.path.join(model_path, 'NM_0_' +
                                      str(job_id*batch_size+i) + inputsuffix +
                                      '.pkl'))
        else:
            print("Extending model ", i+1, "of", feature_num)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + str(i) +
                                      inputsuffix + '.pkl'))

        nm = nm.extend(X, Y[:, i:i+1], batch_effects_train,
                       samples=generation_factor,
                       informative_prior=informative_prior)

        if batch_size is not None:
            nm.save(os.path.join(output_path, 'NM_0_' +
                                 str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            nm.save(os.path.join('Models', 'NM_0_' +
                                 str(i) + outputsuffix + '.pkl'))
        else:
            nm.save(os.path.join(output_path, 'NM_0_' +
                                 str(i) + outputsuffix + '.pkl'))


def tune(covfile, respfile, maskfile=None, **kwargs):
    '''
    This function tunes an existing HBR model with real data.

    Basic usage::

        tune(covfile, respfile [extra_arguments])

    where the variables are defined below.

    :param covfile: covariates for new data
    :param respfile: response variables for new data
    :param maskfile: mask used to apply to the data (nifti only)
    :param model_path: Directory containing the normative model and metadata
    :param trbefile: file address to batch effects file for new data
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param output_path: the path for saving the  the extended model
    :param informative_prior: use initial model prior or learn from scracth (default is False).
    :param generation_factor: see below


    generation factor refers to the number of samples generated for each
    combination of covariates and batch effects. Default is 10.


    All outputs are written to disk in the same format as the input.

    '''

    alg = kwargs.pop('alg')
    if alg != 'hbr':
        print('Model extention is only possible for HBR models.')
        return
    elif (not 'model_path' in list(kwargs.keys())) or \
        (not 'output_path' in list(kwargs.keys())) or \
            (not 'trbefile' in list(kwargs.keys())):
        print('InputError: Some mandatory arguments are missing.')
        return
    else:
        model_path = kwargs.pop('model_path')
        output_path = kwargs.pop('output_path')
        trbefile = kwargs.pop('trbefile')

    outputsuffix = kwargs.pop('outputsuffix', 'tuned')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    informative_prior = kwargs.pop('informative_prior', 'False') == 'True'
    generation_factor = int(kwargs.pop('generation_factor', '10'))
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1

    if not os.path.isdir(model_path):
        print('Models directory does not exist!')
        return
    else:
        if os.path.exists(os.path.join(model_path, 'meta_data.md')):
            with open(os.path.join(model_path, 'meta_data.md'), 'rb') as file:
                meta_data = pickle.load(file)
            if (meta_data['inscaler'] != 'None' or
                    meta_data['outscaler'] != 'None'):
                print('Models extention on scaled data is not possible!')
                return

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # load data
    print("Loading data ...")
    X = fileio.load(covfile)
    Y, maskvol = load_response_vars(respfile, maskfile)
    batch_effects_train = fileio.load(trbefile)

    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    feature_num = Y.shape[1]

    # estimate the models for all subjects
    for i in range(feature_num):

        nm = norm_init(X)
        if batch_size is not None:  # when using nirmative_parallel
            print("Tuning model ", job_id*batch_size+i)
            nm = nm.load(os.path.join(model_path, 'NM_0_' +
                                      str(job_id*batch_size+i) + inputsuffix +
                                      '.pkl'))
        else:
            print("Tuning model ", i+1, "of", feature_num)
            nm = nm.load(os.path.join(model_path, 'NM_0_' + str(i) +
                                      inputsuffix + '.pkl'))

        nm = nm.tune(X, Y[:, i:i+1], batch_effects_train,
                     samples=generation_factor,
                     informative_prior=informative_prior)

        if batch_size is not None:
            nm.save(os.path.join(output_path, 'NM_0_' +
                                 str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            nm.save(os.path.join('Models', 'NM_0_' +
                                 str(i) + outputsuffix + '.pkl'))
        else:
            nm.save(os.path.join(output_path, 'NM_0_' +
                                 str(i) + outputsuffix + '.pkl'))


def merge(covfile=None, respfile=None, **kwargs):
    '''
    This function extends an existing HBR model with data from new sites/scanners.

    Basic usage::

        merge(model_path1, model_path2 [extra_arguments])

    where the variables are defined below.

    :param covfile: Not required. Always set to None.
    :param respfile: Not required. Always set to None.
    :param model_path1: Directory containing the model and metadata (1st model)
    :param model_path2: Directory containing the  model and metadata (2nd model)
    :param batch_size: batch size (for use with normative_parallel)
    :param job_id: batch id
    :param output_path: the path for saving the  the extended model
    :param generation_factor: see below

    The generation factor refers tothe number of samples generated for each
    combination of covariates and batch effects. Default is 10.


    All outputs are written to disk in the same format as the input.

    '''

    alg = kwargs.pop('alg')
    if alg != 'hbr':
        print('Merging models is only possible for HBR models.')
        return
    elif (not 'model_path1' in list(kwargs.keys())) or \
        (not 'model_path2' in list(kwargs.keys())) or \
            (not 'output_path' in list(kwargs.keys())):
        print('InputError: Some mandatory arguments are missing.')
        return
    else:
        model_path1 = kwargs.pop('model_path1')
        model_path2 = kwargs.pop('model_path2')
        output_path = kwargs.pop('output_path')

    outputsuffix = kwargs.pop('outputsuffix', 'merge')
    outputsuffix = "_" + outputsuffix.replace("_", "")
    inputsuffix = kwargs.pop('inputsuffix', 'estimate')
    inputsuffix = "_" + inputsuffix.replace("_", "")
    generation_factor = int(kwargs.pop('generation_factor', '10'))
    job_id = kwargs.pop('job_id', None)
    batch_size = kwargs.pop('batch_size', None)
    if batch_size is not None:
        batch_size = int(batch_size)
        job_id = int(job_id) - 1

    if (not os.path.isdir(model_path1)) or (not os.path.isdir(model_path2)):
        print('Models directory does not exist!')
        return
    else:
        if batch_size is None:
            with open(os.path.join(model_path1, 'meta_data.md'), 'rb') as file:
                meta_data1 = pickle.load(file)
            with open(os.path.join(model_path2, 'meta_data.md'), 'rb') as file:
                meta_data2 = pickle.load(file)
            if meta_data1['valid_voxels'].shape[0] != meta_data2['valid_voxels'].shape[0]:
                print('Two models are trained on different features!')
                return
            else:
                feature_num = meta_data1['valid_voxels'].shape[0]
        else:
            feature_num = batch_size

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # mergeing the models
    for i in range(feature_num):

        nm1 = norm_init(np.random.rand(100, 10))
        nm2 = norm_init(np.random.rand(100, 10))
        if batch_size is not None:  # when using nirmative_parallel
            print("Merging model ", job_id*batch_size+i)
            nm1 = nm1.load(os.path.join(model_path1, 'NM_0_' +
                                        str(job_id*batch_size+i) + inputsuffix +
                                        '.pkl'))
            nm2 = nm2.load(os.path.join(model_path2, 'NM_0_' +
                                        str(job_id*batch_size+i) + inputsuffix +
                                        '.pkl'))
        else:
            print("Merging model ", i+1, "of", feature_num)
            nm1 = nm1.load(os.path.join(model_path1, 'NM_0_' + str(i) +
                                        inputsuffix + '.pkl'))
            nm2 = nm1.load(os.path.join(model_path2, 'NM_0_' + str(i) +
                                        inputsuffix + '.pkl'))

        nm_merged = nm1.merge(nm2, samples=generation_factor)

        if batch_size is not None:
            nm_merged.save(os.path.join(output_path, 'NM_0_' +
                                        str(job_id*batch_size+i) + outputsuffix + '.pkl'))
            nm_merged.save(os.path.join('Models', 'NM_0_' +
                                        str(i) + outputsuffix + '.pkl'))
        else:
            nm_merged.save(os.path.join(output_path, 'NM_0_' +
                                        str(i) + outputsuffix + '.pkl'))


def main(*args):
    """ Parse arguments and estimate model
    """

    np.seterr(invalid='ignore')

    rfile, mfile, cfile, cv, tcfile, trfile, func, alg, cfg, kw = get_args(
        args)

    # collect required arguments
    pos_args = ['cfile', 'rfile']

    # collect basic keyword arguments controlling model estimation
    kw_args = ['maskfile=mfile',
               'cvfolds=cv',
               'testcov=tcfile',
               'testresp=trfile',
               'alg=alg',
               'configparam=cfg']

    # add additional keyword arguments
    for k in kw:
        kw_args.append(k + '=' + "'" + kw[k] + "'")
    all_args = ', '.join(pos_args + kw_args)

    # Executing the target function
    exec(func + '(' + all_args + ')')


# For running from the command line:
if __name__ == "__main__":
    main(sys.argv[1:])

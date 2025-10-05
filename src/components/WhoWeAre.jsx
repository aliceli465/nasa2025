import { useEffect } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from "gsap/all";
import { TiLocationArrow } from "react-icons/ti";

gsap.registerPlugin(ScrollTrigger);

const TeamMember = ({ name, role, image, description }) => {
  return (
    <div className="group relative overflow-hidden rounded-lg border-hsla bg-black/20 backdrop-blur-sm transition-all duration-300 hover:bg-black/40">
      <div className="relative h-64 w-full overflow-hidden">
        <img
          src={image}
          alt={name}
          className="h-full w-full object-cover object-center transition-transform duration-500 group-hover:scale-110"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
      </div>

      <div className="p-6">
        <h3 className="special-font text-xl font-zentry font-thin text-[#dfdff2] uppercase">
          {name}
        </h3>
        <p className="mt-2 font-circular-web text-sm text-[#dfdff2]/70 uppercase tracking-wider">
          {role}
        </p>
        <p className="mt-3 font-circular-web text-sm text-[#dfdff2]/60 leading-relaxed">
          {description}
        </p>
      </div>
    </div>
  );
};

const CreditItem = ({ title, description, link, icon }) => {
  return (
    <div className="group relative h-36 overflow-hidden rounded-lg border-hsla bg-black/20 backdrop-blur-sm p-6 transition-all duration-300 hover:bg-black/40">
      <div className="flex items-start gap-3">
        <div className="mt-1 flex-shrink-0">
          <img
            src={icon}
            alt={`${title} icon`}
            className="h-8 w-8 rounded-lg object-cover object-center"
          />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="special-font text-lg font-zentry font-thin text-[#dfdff2] uppercase">
            {title}
          </h4>
          <p className="mt-2 font-circular-web text-sm text-[#dfdff2]/70 leading-relaxed">
            {description}
          </p>
        </div>
      </div>
      {link && (
        <a
          href={link}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-3 inline-flex items-center gap-2 font-circular-web text-xs text-[#dfdff2]/50 uppercase tracking-wider transition-colors hover:text-[#dfdff2]/80"
        >
          <TiLocationArrow className="text-xs" />
          Visit Source
        </a>
      )}
    </div>
  );
};

const DataSourceItem = ({ title, type, description, authors, year, link }) => {
  const isOurDocumentation = type === "Our Documentation";

  return (
    <div
      className={`group relative overflow-hidden rounded-lg border-hsla backdrop-blur-sm p-5 transition-all duration-300 ${
        isOurDocumentation
          ? "bg-gradient-to-r from-violet-500/20 to-blue-500/20 border-violet-300/30 hover:from-violet-500/30 hover:to-blue-500/30"
          : "bg-black/10 hover:bg-black/20"
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="special-font text-base font-zentry font-thin text-[#dfdff2] uppercase">
            {title}
          </h4>
          <div className="mt-1 flex items-center gap-2">
            <span
              className={`rounded-full px-2 py-1 font-circular-web text-xs uppercase tracking-wider ${
                isOurDocumentation
                  ? "bg-gradient-to-r from-violet-400/30 to-blue-400/30 text-violet-200 font-semibold"
                  : "bg-violet-300/20 text-[#dfdff2]/80"
              }`}
            >
              {type}
            </span>
            <span className="font-circular-web text-xs text-[#dfdff2]/60">
              {year}
            </span>
          </div>
          <p className="mt-2 font-circular-web text-sm text-[#dfdff2]/70 leading-relaxed">
            {description}
          </p>
          <p className="mt-1 font-circular-web text-xs text-[#dfdff2]/60 italic">
            {authors}
          </p>
        </div>
        {link && (
          <a
            href={link}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-3 flex-shrink-0 transition-colors hover:text-[#dfdff2]/80"
          >
            <TiLocationArrow className="text-sm text-[#dfdff2]/50" />
          </a>
        )}
      </div>
    </div>
  );
};

const WhoWeAre = () => {
  // Scroll to top when component mounts
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
  }, []);

  useGSAP(() => {
    // Animate team section
    gsap.fromTo(
      ".team-member",
      {
        opacity: 0,
        y: 50,
        scale: 0.9,
      },
      {
        opacity: 1,
        y: 0,
        scale: 1,
        duration: 0.8,
        stagger: 0.2,
        ease: "power2.out",
        scrollTrigger: {
          trigger: ".team-grid",
          start: "top 80%",
          end: "bottom 20%",
          toggleActions: "play none none reverse",
        },
      }
    );

    // Animate credits section
    gsap.fromTo(
      ".credit-item",
      {
        opacity: 0,
        x: -30,
      },
      {
        opacity: 1,
        x: 0,
        duration: 0.6,
        stagger: 0.1,
        ease: "power2.out",
        scrollTrigger: {
          trigger: ".credits-section",
          start: "top 80%",
          end: "bottom 20%",
          toggleActions: "play none none reverse",
        },
      }
    );

    // Animate data sources section
    gsap.fromTo(
      ".data-source-item",
      {
        opacity: 0,
        y: 30,
      },
      {
        opacity: 1,
        y: 0,
        duration: 0.5,
        stagger: 0.08,
        ease: "power2.out",
        scrollTrigger: {
          trigger: ".data-sources-section",
          start: "top 75%",
          end: "bottom 25%",
          toggleActions: "play none none reverse",
        },
      }
    );

    // Parallax effect for background elements
    gsap.to(".parallax-bg", {
      yPercent: -50,
      ease: "none",
      scrollTrigger: {
        trigger: "main",
        start: "top bottom",
        end: "bottom top",
        scrub: true,
      },
    });
  });

  const teamMembers = [
    {
      name: "Alice Li",
      role: "UI/UX + Frontend",
      image: "/img/alice2.png",
    },
    {
      name: "Allen Chiu",
      role: "Research + Stats",
      image: "/img/allen.jpg",
    },
    {
      name: "Nathan Hu",
      role: "Backend + ML",
      image: "/img/nathan2.jpg",
    },
    {
      name: "William Du",
      role: "Backend + ML",
      image: "/img/william.jpg",
    },
    {
      name: "Zachary Lee",
      role: "Fullstack",
      image: "/img/zach.jpg",
    },
  ];

  const credits = [
    {
      title: "ThreeJS",
      description: "javascript for 3d graphics",
      icon: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhUyPLMCrdBvL7byu5KkMnOssbQigrkiRxZw&s",
    },
    {
      title: "GSAP",
      description: "for all of our scrolling and animation",
      icon: "https://gsap.com/community/uploads/monthly_2020_03/tweenmax.png.cf27916e926fbb328ff214f66b4c8429.png",
    },
    {
      title: "Azure",
      description: "our ML workspace for training our scheduling models",
      icon: "https://ms-toolsai.gallerycdn.vsassets.io/extensions/ms-toolsai/vscode-ai-remote-web/1.0.0/1724367048666/Microsoft.VisualStudio.Services.Icons.Default",
    },
    {
      title: "Veo-3",
      description:
        "for some of our background videos. we promise to support local artists once we become millionares",
      icon: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_ghP0RsA4I4dVWg8zqKlQ0JInX4pFI3QRjA&s",
    },
    {
      title: "Tailwind CSS",
      description: "tailwind, like don crowley, genuinely my goat",
      icon: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Tailwind_CSS_Logo.svg/2560px-Tailwind_CSS_Logo.svg.png",
    },
    {
      title: "Vite",
      description: "deploys react stuff so fast!",
      icon: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vitejs/vitejs-original.svg",
    },
    {
      title: "React",
      description: "of course",
      icon: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/862px-React-icon.svg.png",
    },
    {
      title: "Github",
      description: "of course",
      icon: "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png",
    },
    {
      title: "Python",
      description: "ML language of choice",
      icon: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png",
    },
  ];

  // Removed dataSources array since we're redesigning the section

  return (
    <main className="relative min-h-screen w-full bg-black">
      {/* Background Elements */}
      <div className="parallax-bg absolute inset-0 opacity-5">
        <div className="absolute top-20 left-10 h-96 w-96 rounded-full bg-violet-300 blur-3xl" />
        <div className="absolute bottom-20 right-10 h-80 w-80 rounded-full bg-blue-300 blur-3xl" />
      </div>

      {/* Team Section */}
      <section className="relative z-10 py-16">
        <div className="container mx-auto px-3 md:px-10">
          <div className="mb-20 px-5">
            <h1 className="special-font hero-heading text-[#dfdff2] mb-8">
              Our <b>Te</b>am
            </h1>
          </div>

          <div className="team-grid grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
            {teamMembers.map((member, index) => (
              <div key={index} className="team-member">
                <TeamMember {...member} />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Credits Section */}
      <section className="credits-section relative z-10 bg-black/50 py-16">
        <div className="container mx-auto px-3 md:px-10">
          <div className="mb-8 px-5">
            <h2 className="special-font text-6xl font-zentry font-thin text-[#dfdff2] uppercase md:text-8xl">
              Te<b>ch</b> St<b>ac</b>k
            </h2>
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {credits.map((credit, index) => (
              <div key={index} className="credit-item">
                <CreditItem {...credit} />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Research Documentation Section */}
      <section className="research-docs-section relative z-10 py-16">
        <div className="container mx-auto px-3 md:px-10">
          <div className="flex flex-col lg:flex-row items-center gap-12">
            {/* Left side - Text content */}
            <div className="flex-1 px-5">
              <h2 className="special-font text-6xl font-zentry font-thin text-[#dfdff2] uppercase md:text-8xl mb-8">
                Ou<b>r</b> Th<b>ou</b>gh<b>ts</b>
              </h2>
              <p className="text-xl font-circular-web text-[#dfdff2]/80 leading-relaxed mb-6">
                We've organized all our research methodology, energy
                calculations, and data analysis into one comprehensive PDF
                document. Not an official citation, just something we thought to
                include
              </p>
            </div>

            {/* Right side - PDF Document */}
            <div className="flex-shrink-0">
              <a
                href="/naasa energy calculations.pdf"
                target="_blank"
                rel="noopener noreferrer"
                className="group block"
              >
                <div className="relative bg-gradient-to-br from-violet-500/20 to-blue-500/20 border border-violet-300/30 rounded-xl p-8 hover:from-violet-500/30 hover:to-blue-500/30 transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-violet-500/20">
                  {/* PDF Icon */}
                  <div className="flex items-center justify-center mb-4">
                    <div className="w-16 h-16 bg-red-500 rounded-lg flex items-center justify-center group-hover:bg-red-400 transition-colors">
                      <span className="text-white font-bold text-xl">PDF</span>
                    </div>
                  </div>

                  {/* Click indicator */}
                  <div className="flex items-center justify-center gap-2 text-violet-300 group-hover:text-violet-200 transition-colors">
                    <TiLocationArrow className="text-sm" />
                    <span className="font-circular-web text-xs uppercase tracking-wider">
                      Click to View
                    </span>
                  </div>
                </div>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Special Thanks */}
      <section className="relative z-10 py-16">
        <div className="container mx-auto px-3 md:px-10">
          <div className="mb-8 px-5">
            <h2 className="special-font text-6xl font-zentry font-thin text-[#dfdff2] uppercase md:text-8xl">
              Sp<b>ecial</b> T<b>hanks</b>
            </h2>
          </div>
          <div className="border-hsla rounded-lg bg-black/20 backdrop-blur-sm p-8">
            <p className="text-xl font-circular-web text-[#dfdff2]/80 leading-relaxed mb-6">
              Lastly, special thanks to the entire NASA Space Apps 2025 Chicago
              team for organizing such an amazing event full of so much amazing
              talent.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
};

export default WhoWeAre;

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
  return (
    <div className="group relative overflow-hidden rounded-lg border-hsla bg-black/10 backdrop-blur-sm p-5 transition-all duration-300 hover:bg-black/20">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="special-font text-base font-zentry font-thin text-[#dfdff2] uppercase">
            {title}
          </h4>
          <div className="mt-1 flex items-center gap-2">
            <span className="rounded-full bg-violet-300/20 px-2 py-1 font-circular-web text-xs text-[#dfdff2]/80 uppercase tracking-wider">
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

  const dataSources = [
    {
      title: "GOCE Satellite Data",
      type: "Dataset",
      description:
        "Earth gravitational field measurements for orbital mechanics modeling and satellite positioning algorithms.",
      authors: "European Space Agency (ESA)",
      year: "2009-2013",
      link: "https://www.esa.int/Applications/Observing_the_Earth/The_Living_Planet_Programme/Earth_explorer/GOCE",
    },
    {
      title: "LEO Satellite Constellation Optimization",
      type: "Research Paper",
      description:
        "Machine learning approaches for optimizing satellite constellation deployment and communication scheduling.",
      authors: "Zhang, K. et al.",
      year: "2023",
      link: "https://arxiv.org/abs/2301.04567",
    },
    {
      title: "NASA Space Apps Challenge Dataset",
      type: "Competition Data",
      description:
        "Curated dataset from NASA Space Apps Challenge containing satellite orbital parameters and atmospheric models.",
      authors: "NASA Space Apps Team",
      year: "2025",
      link: "https://spaceappschallenge.org/",
    },
    {
      title: "Starlink Constellation Data",
      type: "Public Dataset",
      description:
        "Historical orbital parameters and constellation performance data for LEO satellite network analysis.",
      authors: "SpaceX",
      year: "2019-Present",
      link: "https://www.spacex.com/starlink",
    },
    {
      title: "Space Environment Data Analysis",
      type: "Research Paper",
      description:
        "Space weather patterns and their impact on satellite communication reliability in LEO environments.",
      authors: "Johnson, M.A. & Chen, L.",
      year: "2024",
      link: "https://doi.org/10.1016/j.ast.2024.105123",
    },
  ];

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

      {/* Data Sources Section */}
      <section className="data-sources-section relative z-10 py-16">
        <div className="container mx-auto px-3 md:px-10">
          <div className="mb-8 px-5">
            <h2 className="special-font text-6xl font-zentry font-thin text-[#dfdff2] uppercase md:text-8xl">
              Scie<b>ntific</b> S<b>ou</b>rces
            </h2>
          </div>

          <div className="space-y-4">
            {dataSources.map((source, index) => (
              <div key={index} className="data-source-item">
                <DataSourceItem {...source} />
              </div>
            ))}
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
            <p className="font-circular-web text-[#dfdff2]/70 leading-relaxed">
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
